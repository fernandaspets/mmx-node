/*
 * EngineWASM.h — WASM JIT execution backend for MMX VM
 *
 * When WITH_WASM_JIT is defined, run_with_wasm_fallback() tries
 * to execute the contract via compiled WebAssembly first.
 * If WASM is unavailable or translation/compilation fails, it falls back
 * to the original interpreter (Engine::run()).
 *
 * Build: cmake -DWITH_WASM_JIT=ON -DWITH_WASM_SHADOW=ON ..
 *   Requires: wasmtime C API (https://wasmtime.dev/)
 *
 * Shadow mode (WITH_WASM_SHADOW): runs both interpreter and WASM,
 * compares results, logs mismatches, but uses interpreter result.
 * This lets you verify WASM correctness on mainnet without risk.
 */

#ifndef INCLUDE_MMX_VM_ENGINE_WASM_H_
#define INCLUDE_MMX_VM_ENGINE_WASM_H_

#include <mmx/vm/Engine.h>

namespace mmx {
namespace vm {

#ifdef WITH_WASM_JIT
	/// Execute contract via WASM JIT. Returns true on success.
	bool run_wasm(Engine& engine);
#endif

	/// Try WASM JIT first, fall back to interpreter. Drop-in for Engine::run().
	void run_with_wasm_fallback(Engine& engine);

} // vm
} // mmx

#endif /* INCLUDE_MMX_VM_ENGINE_WASM_H_ */
