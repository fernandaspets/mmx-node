/*
 * test_wasm_jit.cpp — Benchmark WASM JIT vs interpreter
 *
 * Usage:
 *   ./test_wasm_jit                     — simple ADD test + benchmark
 *   ./test_wasm_jit bytecode.hex         — contract benchmark (no constants)
 *   ./test_wasm_jit bytecode.hex const.hex — contract benchmark with constants
 */

#include <mmx/vm/Engine.h>
#include <mmx/vm/instr_t.h>
#include <mmx/vm/var_t.h>
#include <mmx/vm/WATGenerator.h>
#include <mmx/vm/StorageRAM.h>
#include <mmx/vm/EngineWASM.h>
// deserialize is in var_t.h

#ifdef WITH_WASM_JIT
#include <wasmtime.hh>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <memory>
#include <chrono>

using namespace mmx::vm;

static uint64_t time_ms() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
	std::string clean;
	for(char c : hex) {
		if(c != '\n' && c != '\r' && c != ' ' && c != '\t') clean += c;
	}
	std::vector<uint8_t> bytes;
	for(size_t i = 0; i + 1 < clean.size(); i += 2) {
		bytes.push_back((uint8_t)std::stoi(clean.substr(i, 2), nullptr, 16));
	}
	return bytes;
}

static void load_constants(mmx::vm::Engine& eng, const std::vector<uint8_t>& const_bytes) {
	size_t offset = 0;
	uint64_t dst = 0;
	while(offset < const_bytes.size()) {
		std::unique_ptr<mmx::vm::var_t> var;
		offset += mmx::vm::deserialize(var, const_bytes.data() + offset, const_bytes.size() - offset, false, false);
		if(dst < MEM_EXTERN) {
			eng.assign(dst++, std::move(var));
		}
	}
}

static void test_simple() {
	std::cout << "=== Simple ADD test ===" << std::endl;
	std::vector<instr_t> code = {
		instr_t(OP_ADD, 0, MEM_STACK + 0, MEM_CONST + 1, MEM_CONST + 2),
		instr_t(OP_RET)
	};
	{
		auto storage = std::make_shared<StorageRAM>();
		mmx::vm::Engine engine(mmx::addr_t(), storage, true);
		engine.code = code;
		engine.gas_limit = 100000;
		engine.write(MEM_CONST + 1, uint_t(10));
		engine.write(MEM_CONST + 2, uint_t(20));
		engine.init();
		engine.begin(0);
		engine.run();
		std::cout << "Interpreter: result = " << engine.read_fail<uint_t>(MEM_STACK + 0, TYPE_UINT).value
			<< ", gas = " << engine.gas_used << std::endl;
	}
#ifdef WITH_WASM_JIT
	{
		auto storage = std::make_shared<StorageRAM>();
		mmx::vm::Engine engine(mmx::addr_t(), storage, true);
		engine.code = code;
		engine.gas_limit = 100000;
		engine.write(MEM_CONST + 1, uint_t(10));
		engine.write(MEM_CONST + 2, uint_t(20));
		engine.init();
		engine.begin(0);
		run_with_wasm_fallback(engine);
		std::cout << "WASM JIT: result = " << engine.read_fail<uint_t>(MEM_STACK + 0, TYPE_UINT).value
			<< ", gas = " << engine.gas_used << std::endl;
	}
#endif
}

static void benchmark(const std::vector<instr_t>& code,
		const std::vector<uint8_t>& const_bytes, int iterations) {
	std::cout << "\n=== Benchmark: " << code.size() << " instructions, "
		<< iterations << " iterations ===" << std::endl;

	auto setup = [&](mmx::vm::Engine& eng) {
		eng.code = code;
		eng.gas_limit = 100000000;
		if(!const_bytes.empty()) {
			load_constants(eng, const_bytes);
		}
		eng.init();
		eng.begin(0);
	};

	// Warm up
	{ auto storage = std::make_shared<StorageRAM>(); mmx::vm::Engine eng(mmx::addr_t(), storage, true); setup(eng); eng.run(); }

	auto t0 = time_ms();
	uint64_t interp_gas = 0;
	for(int i = 0; i < iterations; i++) {
		auto storage = std::make_shared<StorageRAM>();
		mmx::vm::Engine eng(mmx::addr_t(), storage, true);
		setup(eng);
		eng.run();
		interp_gas = eng.gas_used;
	}
	auto t1 = time_ms();
	std::cout << "Interpreter: " << (t1 - t0) << " ms ("
		<< (double)(t1 - t0) / iterations << " ms/run, gas=" << interp_gas << ")" << std::endl;

#ifdef WITH_WASM_JIT
	WATGenerator gen(code);
	std::string wat = gen.generate();
	wasmtime::Config config;
	wasmtime::Engine wasm_engine(std::move(config));
	auto compile_result = wasmtime::Module::compile(wasm_engine, wat);
	if(!compile_result) { std::cerr << "WAT compile failed" << std::endl; return; }
	auto module = compile_result.ok();

	// WASM with full setup per iteration (what the node currently does)
	auto t2 = time_ms();
	uint64_t wasm_gas = 0;
	for(int i = 0; i < iterations; i++) {
		auto storage = std::make_shared<StorageRAM>();
		mmx::vm::Engine eng(mmx::addr_t(), storage, true);
		setup(eng);
		try {
			wasmtime::Store store(wasm_engine);
			WASMHostContext ctx;
			ctx.engine = &eng;
			ctx.gas_limit = eng.gas_limit;
			wasmtime::Linker linker(wasm_engine);
			linker.allow_shadowing(true);
			register_host_functions(linker, store, ctx);
			auto inst = linker.instantiate(store, module).ok();
			auto func_opt = inst.get(store.context(), "execute");
			auto func = std::get_if<wasmtime::Func>(&*func_opt);
			func->call(store.context(), {});
			wasm_gas = eng.gas_used;
		} catch(const std::exception& e) {
			if(i == 0) std::cerr << "WASM run failed: " << e.what() << std::endl;
			wasm_gas = eng.gas_used;
		}
	}
	auto t3 = time_ms();
	std::cout << "WASM JIT (full setup):    " << (t3 - t2) << " ms ("
		<< (double)(t3 - t2) / iterations << " ms/run, gas=" << wasm_gas << ")" << std::endl;

	// Now measure just execution time (pre-compiled module, fresh store+linker per call)
	// This is closer to production with module caching
	auto t4 = time_ms();
	for(int i = 0; i < iterations; i++) {
		auto storage = std::make_shared<StorageRAM>();
		mmx::vm::Engine eng(mmx::addr_t(), storage, true);
		setup(eng);
		try {
			wasmtime::Store store(wasm_engine);
			WASMHostContext ctx;
			ctx.engine = &eng;
			ctx.gas_limit = eng.gas_limit;
			wasmtime::Linker linker(wasm_engine);
			linker.allow_shadowing(true);
			register_host_functions(linker, store, ctx);
			auto inst = linker.instantiate(store, module).ok();
			auto func_opt = inst.get(store.context(), "execute");
			auto func = std::get_if<wasmtime::Func>(&*func_opt);
			func->call(store.context(), {});
		} catch(...) {}
	}
	auto t5 = time_ms();
	std::cout << "WASM exec only (no gas check): " << (t5 - t4) << " ms ("
		<< (double)(t5 - t4) / iterations << " ms/run)" << std::endl;

	// Interpreter without setup (just the run)
	auto t6 = time_ms();
	for(int i = 0; i < iterations; i++) {
		auto storage = std::make_shared<StorageRAM>();
		mmx::vm::Engine eng(mmx::addr_t(), storage, true);
		setup(eng);
		eng.run();
	}
	auto t7 = time_ms();
	std::cout << "Interpreter (with setup):   " << (t7 - t6) << " ms ("
		<< (double)(t7 - t6) / iterations << " ms/run)" << std::endl;
	std::cout << "WASM JIT:    " << (t3 - t2) << " ms ("
		<< (double)(t3 - t2) / iterations << " ms/run, gas=" << wasm_gas << ")" << std::endl;
	if(t3 > t2 && t1 > t0) {
		double speedup_full = (double)(t1 - t0) / (t3 - t2);
		std::cout << "Speedup (full setup): " << speedup_full << "x" << std::endl;
	}
	if(t5 > t4 && t7 > t6) {
		double speedup_exec = (double)(t7 - t6) / (t5 - t4);
		std::cout << "Speedup (exec only):  " << speedup_exec << "x" << std::endl;
	}
#endif
}

int main(int argc, char** argv)
{
	if(argc < 2) {
		test_simple();
		std::vector<instr_t> code = {
			instr_t(OP_ADD, 0, MEM_STACK + 0, MEM_CONST + 1, MEM_CONST + 2),
			instr_t(OP_RET)
		};
		// For simple test, write constants manually in setup
		auto simple_setup = [\&](mmx::vm::Engine\& eng) {
			eng.code = code;
			eng.gas_limit = 100000;
			eng.write(MEM_CONST + 1, uint_t(10));
			eng.write(MEM_CONST + 2, uint_t(20));
			eng.init();
			eng.begin(0);
		};
		// Warm up
		{ auto s = std::make_shared<StorageRAM>(); mmx::vm::Engine e(mmx::addr_t(), s, true); simple_setup(e); e.run(); }
		auto t0 = time_ms();
		for(int i = 0; i < 1000; i++) { auto s = std::make_shared<StorageRAM>(); mmx::vm::Engine e(mmx::addr_t(), s, true); simple_setup(e); e.run(); }
		auto t1 = time_ms();
		std::cout << "Interpreter: " << (t1-t0) << " ms (" << (double)(t1-t0)/1000 << " ms/run)" << std::endl;
		return 0;
	}

	std::ifstream bf(argv[1]); std::stringstream bs; bs << bf.rdbuf(); auto byte_bytes = hex_to_bytes(bs.str());
	std::cout << "Bytecode: " << byte_bytes.size() << " bytes" << std::endl;

	std::vector<instr_t> code;
	deserialize(code, byte_bytes.data(), byte_bytes.size());
	std::cout << "Deserialized: " << code.size() << " instructions" << std::endl;

	std::vector<uint8_t> const_bytes;
	if(argc >= 3) {
		std::ifstream cf(argv[2]); std::stringstream cs; cs << cf.rdbuf(); const_bytes = hex_to_bytes(cs.str());
		std::cout << "Constants: " << const_bytes.size() << " bytes" << std::endl;
	}

	int iterations = (argc >= 4) ? std::stoi(argv[3]) : 100;
	benchmark(code, const_bytes, iterations);
	return 0;
}
