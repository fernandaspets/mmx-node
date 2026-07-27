/*
 * EngineWASM.h — WASM JIT execution backend for MMX VM
 *
 * When WITH_WASM_JIT is defined, Engine::run_with_wasm_fallback() tries
 * to execute the contract via compiled WebAssembly first.
 * If WASM is unavailable or translation/compilation fails, it falls back
 * to the original interpreter (Engine::run()).
 *
 * Build: cmake -DWITH_WASM_JIT=ON ..
 *   Requires: libwasmtime-dev (https://wasmtime.dev/)
 *
 * The JS translator (translate.mjs + relooper.mjs) generates WAT from bytecode.
 * In production, either:
 *   a) Embed a JS engine (V8/QuickJS) to run the translator at compile time, or
 *   b) Port the translator to C++ (straightforward — it's mostly string building)
 *
 * The WAT is then compiled to WASM by Wasmtime's built-in WAT parser,
 * and executed via Cranelift JIT compilation for native-speed execution.
 */

#ifndef INCLUDE_MMX_VM_ENGINE_WASM_H_
#define INCLUDE_MMX_VM_ENGINE_WASM_H_

#include <mmx/vm/Engine.h>

namespace mmx {
namespace vm {

#ifdef WITH_WASM_JIT
	/**
	 * Execute contract via WASM JIT.
	 * Returns true if successful, false if should fall back to interpreter.
	 */
	bool run_wasm();
#endif

	/**
	 * Try WASM JIT first, fall back to interpreter.
	 * This is the drop-in replacement for Engine::run().
	 */
	void run_with_wasm_fallback();

} // vm
} // mmx

#endif /* INCLUDE_MMX_VM_ENGINE_WASM_H_ */
