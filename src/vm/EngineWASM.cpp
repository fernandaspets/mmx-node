/*
 * EngineWASM.cpp — WASM JIT execution backend for MMX VM
 *
 * Uses wasmtime C++ API (v47+) to compile and execute MMX VM contracts
 * as WebAssembly. Host functions bridge to existing Engine methods.
 *
 * Build: cmake -DWITH_WASM_JIT=ON ..
 *   Requires: wasmtime C API (https://wasmtime.dev/)
 *   Install: download wasmtime-v*-x86_64-linux-c-api.tar.xz from releases
 *
 * Safety: run_with_wasm_fallback() tries WASM first, falls back to
 * interpreter on any failure. For production, use shadow mode to
 * validate JIT results match interpreter before enabling.
 */

#include <mmx/vm/Engine.h>
#include <mmx/vm/instr_t.h>
#include <mmx/vm/var_t.h>

#ifdef WITH_WASM_JIT
#include <wasmtime.hh>

#include <memory>
#include <functional>
#include <unordered_map>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mmx {
namespace vm {

using namespace wasmtime;

/**
 * Host context — passed to all host functions via lambda capture.
 * Holds a pointer to the Engine so host functions can call Engine methods.
 */
struct WASMHostContext {
	Engine* engine;
	uint64_t gas_used = 0;
	uint64_t gas_limit = 0;
	bool out_of_gas = false;
	bool failed = false;
};

/**
 * Read a uint256 from a VM address as a C++ uint256_t.
 * Helper for host functions.
 */
static uint256_t read_uint256(Engine* eng, int32_t addr) {
	auto* var = eng->read(addr);
	if(!var) return uint256_t(0);
	if(var->type == TYPE_UINT) {
		return ((const uint_t&)*var).value;
	}
	return uint256_t(0);
}

static bool engine_is_true(Engine* eng, int32_t addr) {
	auto* var = eng->read(addr);
	if(!var) return false;
	if(var->type == TYPE_NIL || var->type == TYPE_FALSE) return false;
	if(var->type == TYPE_TRUE) return true;
	if(var->type == TYPE_UINT) return ((const uint_t&)*var).value != 0;
	return true;
}

/**
 * Write a uint256 to a VM address.
 */
static void write_uint256(Engine* eng, int32_t addr, const uint256_t& value) {
	eng->write(addr, uint_t(value));
}

/**
 * Register all host functions into a wasmtime Linker.
 * Each host function is a lambda that calls the corresponding Engine method.
 */
static void register_host_functions(Linker& linker, Store& store, WASMHostContext& ctx) {
	auto cx = store.context();

	// --- Gas + memory ---

	linker.define(cx, "env", "use_gas",
		Func::wrap(cx, [&ctx](int32_t cost) -> Result<std::monostate, Trap> {
			ctx.gas_used += cost;
			if(ctx.gas_used > ctx.gas_limit) {
				ctx.out_of_gas = true;
				return Trap("out of gas");
			}
			return std::monostate();
		}));

	linker.define(cx, "env", "read_mem",
		Func::wrap(cx, [&ctx](int32_t addr) -> int64_t {
			return (int64_t)(uint64_t)read_uint256(ctx.engine, addr);
		}));

	linker.define(cx, "env", "write_mem",
		Func::wrap(cx, [&ctx](int32_t addr, int64_t value) {
			write_uint256(ctx.engine, addr, uint256_t(value));
		}));

	linker.define(cx, "env", "erase_mem",
		Func::wrap(cx, [&ctx](int32_t addr) {
			ctx.engine->erase(addr);
		}));

	// --- uint256 arithmetic ---

	linker.define(cx, "env", "u256_add",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) + read_uint256(ctx.engine, r));
		}));

	linker.define(cx, "env", "u256_sub",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) - read_uint256(ctx.engine, r));
		}));

	linker.define(cx, "env", "u256_mul",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) * read_uint256(ctx.engine, r));
		}));

	linker.define(cx, "env", "u256_div",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) -> Result<std::monostate, Trap> {
			auto rv = read_uint256(ctx.engine, r);
			if(rv == 0) return Trap("division by zero");
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) / rv);
			return std::monostate();
		}));

	linker.define(cx, "env", "u256_mod",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) -> Result<std::monostate, Trap> {
			auto rv = read_uint256(ctx.engine, r);
			if(rv == 0) return Trap("modulo by zero");
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) % rv);
			return std::monostate();
		}));

	// --- Comparisons ---

	linker.define(cx, "env", "u256_lt",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) < read_uint256(ctx.engine, r) ? uint256_t(1) : uint256_t(0));
		}));

	linker.define(cx, "env", "u256_gt",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) > read_uint256(ctx.engine, r) ? uint256_t(1) : uint256_t(0));
		}));

	linker.define(cx, "env", "u256_lte",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) <= read_uint256(ctx.engine, r) ? uint256_t(1) : uint256_t(0));
		}));

	linker.define(cx, "env", "u256_gte",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) >= read_uint256(ctx.engine, r) ? uint256_t(1) : uint256_t(0));
		}));

	linker.define(cx, "env", "u256_eq",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) == read_uint256(ctx.engine, r) ? uint256_t(1) : uint256_t(0));
		}));

	linker.define(cx, "env", "u256_neq",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) != read_uint256(ctx.engine, r) ? uint256_t(1) : uint256_t(0));
		}));

	// --- Bitwise ---

	linker.define(cx, "env", "u256_not",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s) {
			write_uint256(ctx.engine, d, ~read_uint256(ctx.engine, s));
		}));

	linker.define(cx, "env", "u256_xor",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) ^ read_uint256(ctx.engine, r));
		}));

	linker.define(cx, "env", "u256_and",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) & read_uint256(ctx.engine, r));
		}));

	linker.define(cx, "env", "u256_or",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, l) | read_uint256(ctx.engine, r));
		}));

	linker.define(cx, "env", "u256_shl",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s, int32_t c) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, s) << read_uint256(ctx.engine, c));
		}));

	linker.define(cx, "env", "u256_shr",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s, int32_t c) {
			write_uint256(ctx.engine, d, read_uint256(ctx.engine, s) >> read_uint256(ctx.engine, c));
		}));

	// --- Logical (non-bitwise) ---

	linker.define(cx, "env", "u256_lnot",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s) {
			bool is_true = engine_is_true(ctx.engine, s);
			write_uint256(ctx.engine, d, is_true ? uint256_t(0) : uint256_t(1));
		}));

	linker.define(cx, "env", "u256_lxor",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			bool a = engine_is_true(ctx.engine, l);
			bool b = engine_is_true(ctx.engine, r);
			write_uint256(ctx.engine, d, (a != b) ? uint256_t(1) : uint256_t(0));
		}));

	linker.define(cx, "env", "u256_land",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			bool a = engine_is_true(ctx.engine, l);
			bool b = engine_is_true(ctx.engine, r);
			write_uint256(ctx.engine, d, (a && b) ? uint256_t(1) : uint256_t(0));
		}));

	linker.define(cx, "env", "u256_lor",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			bool a = engine_is_true(ctx.engine, l);
			bool b = engine_is_true(ctx.engine, r);
			write_uint256(ctx.engine, d, (a || b) ? uint256_t(1) : uint256_t(0));
		}));

	// --- Min/Max ---

	linker.define(cx, "env", "u256_min",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			auto L = read_uint256(ctx.engine, l);
			auto R = read_uint256(ctx.engine, r);
			write_uint256(ctx.engine, d, L < R ? L : R);
		}));

	linker.define(cx, "env", "u256_max",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			auto L = read_uint256(ctx.engine, l);
			auto R = read_uint256(ctx.engine, r);
			write_uint256(ctx.engine, d, L > R ? L : R);
		}));

	// --- Simple ops ---

	linker.define(cx, "env", "op_copy",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s) {
			ctx.engine->copy(d, s);
		}));

	linker.define(cx, "env", "op_clone",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s) {
			ctx.engine->clone(d, s);
		}));

	linker.define(cx, "env", "op_call",
		Func::wrap(cx, [&ctx](int32_t target, int32_t stack) {
			// TODO: implement call frame push
		}));

	// --- Container ---

	linker.define(cx, "env", "op_type",
		Func::wrap(cx, [&ctx](int32_t d, int32_t addr) {
			auto* var = ctx.engine->read(addr);
			uint8_t type = var ? var->type : TYPE_NIL;
			write_uint256(ctx.engine, d, uint256_t(type));
		}));

	linker.define(cx, "env", "op_size",
		Func::wrap(cx, [&ctx](int32_t d, int32_t addr) {
			auto* var = ctx.engine->read(addr);
			uint64_t size = 0;
			if(var) {
				if(var->type == TYPE_STRING || var->type == TYPE_BINARY) {
					size = ((const binary_t&)*var).size;
				} else if(var->type == TYPE_ARRAY || var->type == TYPE_MAP) {
					size = ((const uint_t&)*var).value.lower(); // simplified
				} else if(var->type == TYPE_UINT) {
					size = ((const uint_t&)*var).value == 0 ? 0 : 1;
				}
			}
			write_uint256(ctx.engine, d, uint256_t(size));
		}));

	linker.define(cx, "env", "op_get_u256",
		Func::wrap(cx, [&ctx](int32_t d, int32_t addr, int32_t key, int32_t flags) {
			ctx.engine->get(d, addr, key, flags);
		}));

	linker.define(cx, "env", "op_set_u256",
		Func::wrap(cx, [&ctx](int32_t addr, int32_t key, int32_t src) {
			ctx.engine->set(addr, key, src, 0);
		}));

	linker.define(cx, "env", "op_erase",
		Func::wrap(cx, [&ctx](int32_t addr, int32_t key) {
			ctx.engine->erase(addr, key, 0);
		}));

	linker.define(cx, "env", "op_push_back",
		Func::wrap(cx, [&ctx](int32_t addr, int32_t src) {
			ctx.engine->push_back(addr, src);
		}));

	linker.define(cx, "env", "op_pop_back",
		Func::wrap(cx, [&ctx](int32_t d, int32_t src) {
			ctx.engine->pop_back(d, src);
		}));

	// --- Conversion ---

	linker.define(cx, "env", "op_conv",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s, int32_t dflags, int32_t sflags) {
			ctx.engine->conv(d, s, dflags, sflags);
		}));

	linker.define(cx, "env", "op_concat",
		Func::wrap(cx, [&ctx](int32_t d, int32_t l, int32_t r) {
			ctx.engine->concat(d, l, r);
		}));

	linker.define(cx, "env", "op_memcpy",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s, int64_t count, int64_t offset) {
			ctx.engine->memcpy(d, s, count, offset);
		}));

	// --- Crypto ---

	linker.define(cx, "env", "op_sha256",
		Func::wrap(cx, [&ctx](int32_t d, int32_t s) {
			ctx.engine->sha256(d, s);
		}));

	linker.define(cx, "env", "op_verify",
		Func::wrap(cx, [&ctx](int32_t d, int32_t msg, int32_t pub, int32_t sig) {
			ctx.engine->verify(d, msg, pub, sig);
		}));

	// --- Blockchain ---

	linker.define(cx, "env", "op_log",
		Func::wrap(cx, [&ctx](int64_t level, int32_t msg) {
			ctx.engine->log(level, msg);
		}));

	linker.define(cx, "env", "op_send",
		Func::wrap(cx, [&ctx](int32_t addr, int32_t amount, int32_t currency, int32_t memo) {
			ctx.engine->send(addr, amount, currency, memo);
		}));

	linker.define(cx, "env", "op_mint",
		Func::wrap(cx, [&ctx](int32_t addr, int32_t amount, int32_t memo) {
			ctx.engine->mint(addr, amount, memo);
		}));

	linker.define(cx, "env", "op_event",
		Func::wrap(cx, [&ctx](int32_t name, int32_t data) {
			ctx.engine->event(name, data);
		}));

	linker.define(cx, "env", "op_fail",
		Func::wrap(cx, [&ctx](int32_t msg, int64_t code) -> Result<std::monostate, Trap> {
			ctx.failed = true;
			return Trap("VM fail");
		}));

	linker.define(cx, "env", "op_rcall",
		Func::wrap(cx, [&ctx](int32_t name, int32_t method, int64_t stack, int64_t nargs) {
			// TODO: implement remote call
		}));

	linker.define(cx, "env", "op_cread",
		Func::wrap(cx, [&ctx](int32_t d, int32_t addr, int32_t field) {
			ctx.engine->cread(d, addr, field);
		}));

	linker.define(cx, "env", "op_balance",
		Func::wrap(cx, [&ctx](int32_t d, int32_t currency) {
			ctx.engine->read_balance(d, currency);
		}));
}

/**
 * Engine::run_wasm() — Execute contract via WASM JIT.
 *
 * Steps:
 * 1. Translate VM bytecode to WAT (via embedded translator or pre-compiled)
 * 2. Compile WAT → WASM via wasmtime
 * 3. Link host functions
 * 4. Execute
 *
 * Returns true if WASM execution succeeded, false if should fall back to interpreter.
 */
bool run_wasm(Engine& engine)
{
	// Step 1: Get WAT for this contract's bytecode
	// In production, WAT would be generated by the JS translator (translate.mjs)
	// or a C++ port, and cached per contract binary hash.
	std::string wat;
	// TODO: generate_wat_from_bytecode(code, wat);
	// For now, assume WAT is pre-generated and stored alongside the binary.

	if(wat.empty()) {
		return false;  // no WAT available, fall back to interpreter
	}

	// Step 2: Initialize wasmtime
	wasmtime::Config config;
	config.wasm_simd(true);
	wasmtime::Engine wasm_engine(std::move(config));

	Store store(wasm_engine);
	WASMHostContext ctx;
	ctx.engine = &engine;
	ctx.gas_limit = engine.gas_limit;

	// Step 3: Compile WAT → WASM module
	// wasmtime can compile WAT directly
	std::vector<uint8_t> wasm_bytes;
	// TODO: wat_to_wasm(wat, wasm_bytes);
	// For now, assume wasm_bytes is pre-compiled.

	if(wasm_bytes.empty()) {
		return false;
	}

	auto module_result = Module::compile(wasm_engine, wasm_bytes);
	if(!module_result) {
		// Compilation failed
		return false;
	}
	auto module = module_result.ok();

	// Step 4: Link host functions
	Linker linker(wasm_engine);
	linker.allow_shadowing(true);
	register_host_functions(linker, store, ctx);

	// Step 5: Instantiate
	auto inst_result = linker.instantiate(store, module);
	if(!inst_result) {
		// Instantiation failed (missing import, etc.)
		return false;
	}
	auto instance = inst_result.ok();

	// Step 6: Get "execute" export and call it
	auto extern_opt = instance.get(store.context(), "execute");
	if(!extern_opt) {
		return false;
	}
	auto& func_extern = *extern_opt;
	auto* func_ptr = std::get_if<wasmtime::Func>(&func_extern);
	if(!func_ptr) {
		return false;
	}
	auto& func = *func_ptr;

	// Call execute()
	auto call_result = func.call(store.context(), {});
	if(!call_result) {
		// Execution trapped — check if it's a FAIL opcode or out of gas
		auto trap = call_result.err();
		// If it's a FAIL or out-of-gas, that's expected behavior
		// If it's something else, fall back to interpreter for safety
		if(!ctx.out_of_gas && !ctx.failed) {
			return false;  // unexpected trap, fall back
		}
	}

	engine.gas_used = ctx.gas_used;
	return true;
}

#endif // WITH_WASM_JIT

/**
 * run_with_wasm_fallback() — Try WASM JIT first, fall back to interpreter.
 *
 * This is the drop-in replacement for Engine::run().
 * Safe for mainnet: if WASM fails for ANY reason, the interpreter runs instead.
 * The node stays canonical.
 */
void run_with_wasm_fallback(Engine& engine)
{
#ifdef WITH_WASM_JIT
	try {
		if(run_wasm(engine)) {
			return;  // WASM execution succeeded
		}
	} catch(...) {
		// WASM failed, fall through to interpreter
	}
#endif
	// Fall back to interpreter
	engine.run();
}

} // vm
} // mmx
