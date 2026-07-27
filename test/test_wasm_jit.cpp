/*
 * test_wasm_jit.cpp — Test WASM JIT with real contract bytecode
 *
 * 1. Deserialize TRAIL contract binary (from chain)
 * 2. Generate WAT via WATGenerator
 * 3. Compile WAT → WASM via wasmtime
 * 4. Execute with host functions bridging to Engine
 * 5. Compare results with interpreter
 *
 * Build: part of mmx_vm_tests with WITH_WASM_JIT=ON
 */
#include <algorithm>

#include <mmx/vm/Engine.h>
#include <mmx/vm/instr_t.h>
#include <mmx/vm/var_t.h>
#include <mmx/vm/WATGenerator.h>
#include <mmx/vm/StorageRAM.h>

#ifdef WITH_WASM_JIT
#include <wasmtime.hh>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <memory>

using namespace mmx::vm;

int main(int argc, char** argv)
{
	// Step 1: Load bytecode
	std::string hex;
	if(argc > 1) {
		std::ifstream file(argv[1]);
		if(!file.is_open()) {
			std::cerr << "Failed to open " << argv[1] << std::endl;
			return 1;
		}
		std::stringstream ss;
		ss << file.rdbuf();
		hex = ss.str();
		// Remove whitespace/newlines
		std::string clean;
		for(char c : hex) {
		if(c != '\n' && c != '\r' && c != ' ' && c != '\t') {
			clean += c;
		}
	}
		hex = clean;
	} else {
		// Use a simple test program: ADD 10 + 20 = 30
		std::cout << "No bytecode file provided, using simple test program." << std::endl;
		std::vector<instr_t> test_code = {
			instr_t(OP_ADD, 0, 0, 1, 2),   // addr[0] = addr[1] + addr[2]
			instr_t(OP_RET, 0, 0, 0, 0)
		};

		// Generate WAT
		WATGenerator gen(test_code);
		std::string wat = gen.generate();
		std::cout << "Generated WAT (" << wat.size() << " bytes):" << std::endl;
		std::cout << wat << std::endl;

#ifdef WITH_WASM_JIT
		// Try compiling via wasmtime
		try {
			wasmtime::Config config;
			wasmtime::Engine engine(std::move(config));
			wasmtime::Store store(engine);

			auto result = wasmtime::Module::compile(engine, wat);
			if(!result) {
				std::cerr << "WAT compilation failed!" << std::endl;
				return 1;
			}
			auto module = result.ok();
			std::cout << "WAT compiled to WASM successfully!" << std::endl;
			std::cout << "Module compiled, ready for execution." << std::endl;
		} catch(const std::exception& e) {
			std::cerr << "Error: " << e.what() << std::endl;
			return 1;
		}
#else
		std::cout << "WASM JIT not enabled (WITH_WASM_JIT=OFF)" << std::endl;
#endif
		return 0;
	}

	// Deserialize hex bytecode
	std::vector<uint8_t> bytes;
	for(size_t i = 0; i + 1 < hex.size(); i += 2) {
		auto byte = std::stoi(hex.substr(i, 2), nullptr, 16);
		bytes.push_back((uint8_t)byte);
	}

	std::cout << "Bytecode: " << bytes.size() << " bytes" << std::endl;

	std::vector<instr_t> code;
	try {
		deserialize(code, bytes.data(), bytes.size());
	} catch(const std::exception& e) {
		std::cerr << "Deserialization failed: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "Deserialized: " << code.size() << " instructions" << std::endl;

	// Generate WAT
	WATGenerator gen(code);
	std::string wat = gen.generate();
	size_t wat_lines = std::count(wat.begin(), wat.end(), '\n') + 1;
	std::cout << "Generated WAT: " << wat.size() << " bytes, " << wat_lines << " lines" << std::endl;

#ifdef WITH_WASM_JIT
	// Compile via wasmtime
	try {
		wasmtime::Config config;
		wasmtime::Engine engine(std::move(config));
		wasmtime::Store store(engine);

		std::cout << "Compiling WAT → WASM..." << std::endl;
		auto result = wasmtime::Module::compile(engine, wat);
		if(!result) {
			std::cerr << "WAT compilation failed!" << std::endl;
			// Print first few lines of WAT for debugging
			std::istringstream iss(wat);
			std::string line;
			int n = 0;
			while(std::getline(iss, line) && n++ < 10) {
				std::cerr << "  " << line << std::endl;
			}
			return 1;
		}
		auto module = result.ok();
		std::cout << "WAT compiled to WASM successfully!" << std::endl;

		// TODO: set up Engine with storage, register host functions, instantiate, execute
		std::cout << "Module ready for execution (host function linking TODO)." << std::endl;

	} catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
#else
	std::cout << "WASM JIT not enabled. WAT generated but not compiled." << std::endl;
#endif

	return 0;
}
