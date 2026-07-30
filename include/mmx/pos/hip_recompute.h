/*
 * hip_recompute.h — AMD GPU (HIP/ROCm) proof recompute interface
 * Same API as cuda_recompute.h but for AMD GPUs via HIP
 */
#ifndef INCLUDE_MMX_POS_HIP_RECOMPUTE_H_
#define INCLUDE_MMX_POS_HIP_RECOMPUTE_H_

#include <mmx/hash_t.hpp>
#include <mmx/pos/config.h>

#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include <memory>

namespace mmx {
namespace pos {

struct hip_device_t {
	int index = -1;
	std::string name;
	uint32_t max_resident = 0;
	uint64_t buffer_size = 0;
};

struct hip_result_t {
	uint64_t id = 0;
	bool failed = false;
	std::string error;
	std::vector<uint32_t> X;
	std::vector<std::pair<uint32_t, bytes_t<META_BYTES_OUT>>> entries;
};

bool have_hip_recompute();

std::vector<hip_device_t> get_hip_devices();

std::vector<hip_device_t> get_hip_devices_used();

void hip_recompute_init(bool enable = true, std::vector<int> device_list = {});

void hip_recompute_shutdown();

uint64_t hip_recompute(const int ksize, const int xbits, const hash_t& plot_id, const std::vector<uint32_t>& x_values);

std::shared_ptr<const hip_result_t> hip_recompute_poll(const std::set<uint64_t>& jobs);

} // pos
} // mmx

#endif /* INCLUDE_MMX_POS_HIP_RECOMPUTE_H_ */
