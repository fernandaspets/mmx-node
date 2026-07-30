# HIP/ROCm AMD GPU Proof Recompute

Port of `cuda_recompute.cu` to HIP for AMD GPUs. Same API, same kernels, with AMD-specific fixes.

## What was built

**6 new files:**
- `include/hip_runtime_util.h` — HIP error checking wrapper
- `include/hip_util.h` — device-side helpers (bswap, rotl)
- `include/hip_sha512.h` — SHA-512 with circular buffer fix + memory clobber barriers
- `include/mmx/pos/hip_recompute.h` — public API (mirrors `cuda_recompute.h`)
- `src/pos/hip_recompute.hip` — full port of `cuda_recompute.cu` (634 lines)

**5 modified files:**
- `CMakeLists.txt` — `WITH_HIP` option, auto-detect arch, `HIP_ENABLE_WARP_SYNC_BUILTINS`
- `src/pos/verify.cpp` — `WITH_HIP` block after `WITH_CUDA` block
- `src/mmx_node.cpp`, `mmx_harvester.cpp`, `mmx_farmer.cpp` — init/shutdown wired

## AMD-specific fixes applied (from mmx-rocm-plotter)

1. **SHA-512 circular buffer** — `w[80]` → `w[16]` with modular indexing (HIP private memory limit ~1KB, full array silently corrupts hash output)
2. **Memory clobber barriers** — `asm volatile` for type-pun safety (uint32→uint64 cast invisible to HIP/clang -O2)
3. **`__syncwarp()` → `__syncthreads()`** — ROCm 6.x doesn't have `__syncwarp`
4. **`__shfl_sync` ULL masks** — HIP requires `unsigned long long` (0xFFFFFFFFULL, not 0xFFFFFFFF)
5. **`hipMallocHost` cast** — requires `(void**)` unlike CUDA's typed pointers

## Build

```bash
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DWITH_HIP=ON -DGPU_ARCH=gfx1100 ..
ninja -j$(nproc) mmx_node
```

GPU arch auto-detected via `rocminfo`. Override with `-DGPU_ARCH=gfx1030` etc.

## Tested

ROCm 6.4.2, gfx1100 (RX 7900 XTX). Binary links `libamdhip64.so.6` and contains `have_hip_recompute` / `hip_recompute_init` symbols.

AMD farmers can now use GPU-accelerated proof recompute instead of CPU fallback.

## Credits

- Original CUDA implementation: [madMAx43v3r](https://github.com/madMAx43v3r/mmx-node)
- SHA-512 circular buffer + memory clobber fixes: [mmx-rocm-plotter](https://github.com/laxracket/mmx-rocm-plotter) by laxracket
