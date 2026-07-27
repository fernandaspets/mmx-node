#ifndef INCLUDE_MMX_VM_WATGENERATOR_H_
#define INCLUDE_MMX_VM_WATGENERATOR_H_

#include <mmx/vm/instr_t.h>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <cstdint>

namespace mmx {
namespace vm {

struct LoopRange {
	size_t loop_end;
	size_t exit_target;
};

struct EmitResult {
	std::vector<std::string> lines;
	size_t next_index;
};

class WATGenerator {
public:
	explicit WATGenerator(const std::vector<instr_t>& code);
	std::string generate();

private:
	const std::vector<instr_t>& code_;
	int label_count_ = 0;
	std::map<size_t, LoopRange> loops_;

	std::string next_label();
	void find_loop_ranges();
	EmitResult emit_structured(size_t start, size_t end, int64_t skip_loop_start = -1);
	void emit_instruction(const instr_t& inst, std::vector<std::string>& lines);
	void emit_call2(std::vector<std::string>& lines, const char* name,
		uint32_t a, uint32_t b, uint64_t gas);
	void emit_call3(std::vector<std::string>& lines, const char* name,
		uint32_t a, uint32_t b, uint32_t c, uint64_t gas);
	static std::string to_hex(uint8_t v);
	void emit_module(std::ostringstream& out);
	void emit_imports(std::ostringstream& out);
};

} // vm
} // mmx

#endif
