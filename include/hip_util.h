/*
 * hip_util.h — Device-side utility functions for HIP/ROCm
 * Adapted from cuda_util.h + rocm_util.h (from mmx-rocm-plotter)
 */
#ifndef INCLUDE_MMX_HIP_UTIL_H_
#define INCLUDE_MMX_HIP_UTIL_H_

#include <hip/hip_runtime.h>
#include <cstdint>

typedef unsigned long long int uint64_gpu;

__device__ inline
uint32_t hip_bswap_32(const uint32_t y) {
	return (y << 24) | ((y << 8) & 0xFF0000) | ((y >> 8) & 0xFF00) | (y >> 24);
}

__device__ inline
uint64_gpu hip_bswap_64(const uint64_gpu y) {
	return (uint64_gpu(hip_bswap_32(y)) << 32) | hip_bswap_32(y >> 32);
}

__device__ inline
uint32_t hip_rotl_32(const uint32_t w, const uint32_t c) {
	return __funnelshift_l(w, w, c);
}

#endif /* INCLUDE_MMX_HIP_UTIL_H_ */
