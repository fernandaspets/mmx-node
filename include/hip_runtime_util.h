/*
 * hip_runtime_util.h — HIP runtime helpers for AMD GPU support
 * Minimal wrapper for MMX node (only what recompute needs)
 */
#ifndef INCLUDE_MMX_HIP_RUNTIME_UTIL_H_
#define INCLUDE_MMX_HIP_RUNTIME_UTIL_H_

#include <string>
#include <stdexcept>
#include <hip/hip_runtime.h>

inline void hip_check(const hipError_t& code, const std::string& message = std::string()) {
	if(code != hipSuccess) {
		throw std::runtime_error("HIP error " + std::to_string(code) + ": " + message + std::string(hipGetErrorString(code)));
	}
}

#endif /* INCLUDE_MMX_HIP_RUNTIME_UTIL_H_ */
