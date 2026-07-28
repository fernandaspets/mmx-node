/*
 * WATGenerator.cpp — C++ port of translate.mjs + relooper.mjs
 *
 * Generates WebAssembly Text Format (WAT) from MMX VM bytecode.
 * The WAT is then compiled to WASM by wasmtime and executed via Cranelift JIT.
 *
 * This is a direct port of the JS translator. It handles:
 *   - All 48 opcodes → host function calls
 *   - Relooper: arbitrary jumps → structured WASM block/loop/br_if
 *   - if, if-else, while, for patterns
 *   - Gas metering via use_gas host call
 */

#include <mmx/vm/instr_t.h>
#include <mmx/vm/var_t.h>
#include <mmx/vm/WATGenerator.h>

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <cstdint>

namespace mmx {
namespace vm {

// ============================================================
// Opcode constants (from instr_t.h)
// ============================================================

using opcode_e = ::mmx::vm::opcode_e;

// ============================================================
// Gas costs (from Engine.h)
// ============================================================

static constexpr uint64_t GAS_INSTR = 20;
static constexpr uint64_t GAS_CALL = 30;
static constexpr uint64_t GAS_MUL_256 = 200;
static constexpr uint64_t GAS_DIV_256 = 5000;
static constexpr uint64_t GAS_WRITE = 50;
static constexpr uint64_t GAS_STOR_READ = 2000;
static constexpr uint64_t GAS_STOR_WRITE = 2000;
static constexpr uint64_t GAS_SHA256 = 2000;
static constexpr uint64_t GAS_ECDSA = 20000;

// ============================================================
// Helpers
// ============================================================

static bool is_jump(uint8_t code) {
	return code == OP_JUMP || code == OP_JUMPI || code == OP_JUMPN;
}

// ============================================================
// Relooper
// ============================================================

// WATGenerator class definition is in WATGenerator.h
// Method definitions follow below.

std::string WATGenerator::next_label() {
	return "l" + std::to_string(label_count_++);
}

void WATGenerator::find_loop_ranges() {
	loops_.clear();
		for(size_t i = 0; i < code_.size(); i++) {
			const auto& inst = code_[i];
			if(is_jump(inst.code) && inst.a <= i) {
				size_t loop_start = inst.a;
				size_t loop_end = i;
				size_t exit_target = loop_end + 1;
				for(size_t j = loop_start; j < loop_end; j++) {
					if(code_[j].code == OP_JUMPN && code_[j].a > loop_end) {
						exit_target = code_[j].a;
						break;
					}
				}
				loops_[loop_start] = {loop_end, exit_target};
			}
		}
	}

EmitResult WATGenerator::emit_structured(size_t start, size_t end, int64_t skip_loop_start) {
		std::vector<std::string> lines;
		size_t i = start;

		while(i < end) {
			const auto& inst = code_[i];

			// === Loop detection ===
			if((int64_t)i != skip_loop_start && loops_.count(i)) {
				const auto& lr = loops_[i];
				std::string loop_label = next_label();
				std::string exit_label = next_label();

				// Find the JUMPN that exits the loop
				int jumnp_idx = -1;
				for(size_t j = i; j < lr.loop_end; j++) {
					if(code_[j].code == OP_JUMPN && code_[j].a > lr.loop_end) {
						jumnp_idx = (int)j;
						break;
					}
				}

				lines.push_back("    (block $" + exit_label);
				lines.push_back("      (loop $" + loop_label);

				if(jumnp_idx >= 0) {
					uint32_t cond_addr = code_[jumnp_idx].b;
					// Emit condition computation [i..jumnp_idx-1]
					for(size_t j = i; j < (size_t)jumnp_idx; j++) {
						const auto& ci = code_[j];
						if(!is_jump(ci.code) && ci.code != OP_RET && ci.code != OP_CALL) {
							emit_instruction(ci, lines);
						}
					}
					lines.push_back("        (br_if $" + exit_label +
						" (i64.eqz (call $read_mem (i32.const " + std::to_string(cond_addr) + "))))");
					// Emit loop body [jumnp_idx+1..loop_end-1]
					auto body = emit_structured(jumnp_idx + 1, lr.loop_end, (int64_t)i);
					for(const auto& l : body.lines) lines.push_back(l);
				} else {
					auto body = emit_structured(i, lr.loop_end, (int64_t)i);
					for(const auto& l : body.lines) lines.push_back(l);
				}

				lines.push_back("        (br $" + loop_label + ")");
				lines.push_back("      )");
				lines.push_back("    )");
				i = lr.exit_target;
				continue;
			}

			// === Forward JUMPN (if-block) ===
			if(inst.code == OP_JUMPN && inst.a > i) {
				size_t target = inst.a;
				uint32_t cond_addr = inst.b;

				// Check for if-else (forward JUMP inside)
				bool has_forward_jump = false;
				int forward_jump_idx = -1;
				size_t forward_jump_target = 0;
				for(size_t j = i + 1; j < target; j++) {
					if(code_[j].code == OP_JUMP && code_[j].a > j) {
						has_forward_jump = true;
						forward_jump_idx = (int)j;
						forward_jump_target = code_[j].a;
						break;
					}
				}

				if(has_forward_jump) {
					// if-else pattern
					std::string if_label = next_label();
					std::string after_label = next_label();
					lines.push_back("    (block $" + after_label);
					lines.push_back("      (block $" + if_label);
					lines.push_back("        (br_if $" + if_label +
						" (i64.eqz (call $read_mem (i32.const " + std::to_string(cond_addr) + "))))");
					// Emit if-body [i+1..forward_jump_idx)
					for(size_t j = i + 1; j < (size_t)forward_jump_idx; j++) {
						if(!is_jump(code_[j].code) && code_[j].code != OP_RET && code_[j].code != OP_CALL) {
							emit_instruction(code_[j], lines);
						}
					}
					lines.push_back("        (br $" + after_label + ")");
					lines.push_back("      )");
					// Emit else-body [forward_jump_idx+1..forward_jump_target)
					auto else_body = emit_structured(forward_jump_idx + 1, forward_jump_target, skip_loop_start);
					for(const auto& l : else_body.lines) lines.push_back(l);
					lines.push_back("    )");
					i = std::max(forward_jump_target, else_body.next_index);
				} else {
					// Simple if (no else)
					std::string if_label = next_label();
					lines.push_back("    (block $" + if_label);
					lines.push_back("      (br_if $" + if_label +
						" (i64.eqz (call $read_mem (i32.const " + std::to_string(cond_addr) + "))))");
					auto body = emit_structured(i + 1, target, skip_loop_start);
					for(const auto& l : body.lines) lines.push_back(l);
					lines.push_back("    )");
					i = std::max(target, body.next_index);
				}
				continue;
			}

			// === Forward JUMPI (jump if true) ===
			if(inst.code == OP_JUMPI && inst.a > i) {
				size_t target = inst.a;
				uint32_t cond_addr = inst.b;
				std::string if_label = next_label();
				lines.push_back("    (block $" + if_label);
				lines.push_back("      (br_if $" + if_label +
					" (i32.ne (i64.eqz (call $read_mem (i32.const " + std::to_string(cond_addr) + "))) (i32.const 0)))");
				auto body = emit_structured(i + 1, target, skip_loop_start);
				for(const auto& l : body.lines) lines.push_back(l);
				lines.push_back("    )");
				i = std::max(target, body.next_index);
				continue;
			}

			// === Forward JUMP (unconditional skip) ===
			if(inst.code == OP_JUMP && inst.a > i) {
				size_t target = inst.a;
				std::string skip_label = next_label();
				lines.push_back("    (block $" + skip_label);
				lines.push_back("      (br $" + skip_label + ")");
				auto body = emit_structured(i + 1, target, skip_loop_start);
				for(const auto& l : body.lines) lines.push_back(l);
				lines.push_back("    )");
				i = std::max(target, body.next_index);
				continue;
			}

			// === Backward jump (handled by loop) ===
			if(is_jump(inst.code) && inst.a <= i) {
				lines.push_back("    ;; backward jump to " + std::to_string(inst.a));
				i++;
				continue;
			}

			// === RET ===
			if(inst.code == OP_RET) {
				lines.push_back("    (call $use_gas (i32.const 20))");
				lines.push_back("    (return (call $read_mem (i32.const 0)))");
				i++;
				continue;
			}

			// === CALL ===
			if(inst.code == OP_CALL) {
				lines.push_back("    (call $op_call (i32.const " + std::to_string(inst.a) +
					") (i32.const " + std::to_string(inst.b) + "))");
				lines.push_back("    (call $use_gas (i32.const 50))");
				i++;
				continue;
			}

			// === Normal instruction ===
			emit_instruction(inst, lines);
			i++;
		}

		return {lines, i};
	}

	// ============================================================
	// Emit a single instruction as WAT
	// ============================================================

void WATGenerator::emit_instruction(const instr_t& inst, std::vector<std::string>& lines) {
		const auto a = inst.a;
		const auto b = inst.b;
		const auto c = inst.c;
		const auto d = inst.d;
		const auto flags = inst.flags;

		switch(inst.code) {
			// === Simple ops ===
			case OP_NOP:
				lines.push_back("    (call $use_gas (i32.const 20))");
				break;
			case OP_CLR:
				lines.push_back("    (call $erase_mem (i32.const " + std::to_string(a) + "))");
				lines.push_back("    (call $use_gas (i32.const 20))");
				break;
			case OP_COPY:
				lines.push_back("    (call $op_copy (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + "))");
				lines.push_back("    (call $use_gas (i32.const 20))");
				break;
			case OP_CLONE:
				lines.push_back("    (call $op_clone (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + "))");
				lines.push_back("    (call $use_gas (i32.const 70))");
				break;

			// === Arithmetic ===
			case OP_ADD:
				emit_call3(lines, "u256_add", a, b, c, GAS_INSTR);
				break;
			case OP_SUB:
				emit_call3(lines, "u256_sub", a, b, c, GAS_INSTR);
				break;
			case OP_MUL:
				emit_call3(lines, "u256_mul", a, b, c, GAS_INSTR + GAS_MUL_256);
				break;
			case OP_DIV:
				emit_call3(lines, "u256_div", a, b, c, GAS_INSTR + GAS_DIV_256);
				break;
			case OP_MOD:
				emit_call3(lines, "u256_mod", a, b, c, GAS_INSTR + GAS_DIV_256);
				break;

			// === Bitwise ===
			case OP_NOT:
				if(flags & OPFLAG_BITWISE) {
					emit_call2(lines, "u256_not", a, b, GAS_INSTR);
				} else {
					emit_call2(lines, "u256_lnot", a, b, GAS_INSTR);
				}
				break;
			case OP_XOR:
				if(flags & OPFLAG_BITWISE) {
					emit_call3(lines, "u256_xor", a, b, c, GAS_INSTR);
				} else {
					emit_call3(lines, "u256_lxor", a, b, c, GAS_INSTR);
				}
				break;
			case OP_AND:
				if(flags & OPFLAG_BITWISE) {
					emit_call3(lines, "u256_and", a, b, c, GAS_INSTR);
				} else {
					emit_call3(lines, "u256_land", a, b, c, GAS_INSTR);
				}
				break;
			case OP_OR:
				if(flags & OPFLAG_BITWISE) {
					emit_call3(lines, "u256_or", a, b, c, GAS_INSTR);
				} else {
					emit_call3(lines, "u256_lor", a, b, c, GAS_INSTR);
				}
				break;
			case OP_MIN:
				emit_call3(lines, "u256_min", a, b, c, GAS_INSTR);
				break;
			case OP_MAX:
				emit_call3(lines, "u256_max", a, b, c, GAS_INSTR);
				break;
			case OP_SHL:
				emit_call3(lines, "u256_shl", a, b, c, GAS_INSTR);
				break;
			case OP_SHR:
				emit_call3(lines, "u256_shr", a, b, c, GAS_INSTR);
				break;

			// === Comparisons ===
			case OP_CMP_EQ:
				emit_call3(lines, "u256_eq", a, b, c, GAS_INSTR);
				break;
			case OP_CMP_NEQ:
				emit_call3(lines, "u256_neq", a, b, c, GAS_INSTR);
				break;
			case OP_CMP_LT:
				emit_call3(lines, "u256_lt", a, b, c, GAS_INSTR);
				break;
			case OP_CMP_GT:
				emit_call3(lines, "u256_gt", a, b, c, GAS_INSTR);
				break;
			case OP_CMP_LTE:
				emit_call3(lines, "u256_lte", a, b, c, GAS_INSTR);
				break;
			case OP_CMP_GTE:
				emit_call3(lines, "u256_gte", a, b, c, GAS_INSTR);
				break;

			// === Container ===
			case OP_TYPE:
				emit_call2(lines, "op_type", a, b, GAS_INSTR);
				break;
			case OP_SIZE:
				emit_call2(lines, "op_size", a, b, GAS_INSTR);
				break;
			case OP_GET:
				lines.push_back("    (call $op_get_u256 (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (i32.const " + std::to_string(c) +
					") (i32.const " + std::to_string(flags) + "))");
				lines.push_back("    (call $use_gas (i32.const " + std::to_string(GAS_INSTR + GAS_STOR_READ) + "))");
				break;
			case OP_SET:
				lines.push_back("    (call $op_set_u256 (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (i32.const " + std::to_string(c) + "))");
				lines.push_back("    (call $use_gas (i32.const " + std::to_string(GAS_INSTR + GAS_STOR_WRITE) + "))");
				break;
			case OP_ERASE:
				emit_call2(lines, "op_erase", a, b, GAS_INSTR);
				break;
			case OP_PUSH_BACK:
				emit_call2(lines, "op_push_back", a, b, GAS_INSTR + GAS_WRITE);
				break;
			case OP_POP_BACK:
				emit_call2(lines, "op_pop_back", a, b, GAS_INSTR);
				break;

			// === Conversion ===
			case OP_CONV:
				lines.push_back("    (call $op_conv (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (i32.const " + std::to_string(c) +
					") (i32.const " + std::to_string(d) + "))");
				lines.push_back("    (call $use_gas (i32.const 20))");
				break;
			case OP_CONCAT:
				emit_call3(lines, "op_concat", a, b, c, GAS_INSTR);
				break;
			case OP_MEMCPY:
				lines.push_back("    (call $op_memcpy (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (call $read_mem (i32.const " +
					std::to_string(c) + ")) (call $read_mem (i32.const " + std::to_string(d) + ")))");
				lines.push_back("    (call $use_gas (i32.const 20))");
				break;

			// === Crypto ===
			case OP_SHA256:
				emit_call2(lines, "op_sha256", a, b, GAS_INSTR + GAS_SHA256);
				break;
			case OP_VERIFY:
				lines.push_back("    (call $op_verify (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (i32.const " + std::to_string(c) + ") (i32.const " + std::to_string(d) + "))");
				lines.push_back("    (call $use_gas (i32.const " + std::to_string(GAS_INSTR + GAS_ECDSA) + "))");
				break;

			// === Blockchain ===
			case OP_LOG:
				lines.push_back("    (call $op_log (call $read_mem (i32.const " + std::to_string(a) +
					")) (i32.const " + std::to_string(b) + "))");
				lines.push_back("    (call $use_gas (i32.const 20))");
				break;
			case OP_SEND:
				lines.push_back("    (call $op_send (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (i32.const " + std::to_string(c) +
					") (i32.const " + std::to_string(d) + "))");
				lines.push_back("    (call $use_gas (i32.const 20))");
				break;
			case OP_MINT:
				lines.push_back("    (call $op_mint (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (i32.const " + std::to_string(c) + "))");
				lines.push_back("    (call $use_gas (i32.const 20))");
				break;
			case OP_EVENT:
				emit_call2(lines, "op_event", a, b, GAS_INSTR);
				break;
			case OP_FAIL:
				lines.push_back("    (call $op_fail (i32.const " + std::to_string(a) +
					") (call $read_mem (i32.const " + std::to_string(b) + ")))");
				lines.push_back("    (call $use_gas (i32.const 20))");
				lines.push_back("    (return (i64.const -1))");
				break;
			case OP_RCALL:
				lines.push_back("    (call $op_rcall (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (call $read_mem (i32.const " +
					std::to_string(c) + ")) (call $read_mem (i32.const " + std::to_string(d) + ")))");
				lines.push_back("    (call $use_gas (i32.const " + std::to_string(GAS_INSTR + GAS_STOR_READ) + "))");
				break;
			case OP_CREAD:
				lines.push_back("    (call $op_cread (i32.const " + std::to_string(a) +
					") (i32.const " + std::to_string(b) + ") (i32.const " + std::to_string(c) + "))");
				lines.push_back("    (call $use_gas (i32.const " + std::to_string(GAS_INSTR + GAS_STOR_READ) + "))");
				break;
			case OP_BALANCE:
				emit_call2(lines, "op_balance", a, b, GAS_INSTR + GAS_STOR_READ);
				break;

			default:
				lines.push_back("    ;; UNKNOWN opcode 0x" + to_hex(inst.code));
				lines.push_back("    (call $use_gas (i32.const 20))");
		}
	}

	// ============================================================
	// Emit helpers
	// ============================================================

void WATGenerator::emit_call2(std::vector<std::string>& lines, const char* name,
			uint32_t a, uint32_t b, uint64_t gas) {
		lines.push_back("    (call $" + std::string(name) + " (i32.const " + std::to_string(a) +
			") (i32.const " + std::to_string(b) + "))");
		lines.push_back("    (call $use_gas (i32.const " + std::to_string(gas) + "))");
	}

void WATGenerator::emit_call3(std::vector<std::string>& lines, const char* name,
			uint32_t a, uint32_t b, uint32_t c, uint64_t gas) {
		lines.push_back("    (call $" + std::string(name) + " (i32.const " + std::to_string(a) +
			") (i32.const " + std::to_string(b) + ") (i32.const " + std::to_string(c) + "))");
		lines.push_back("    (call $use_gas (i32.const " + std::to_string(gas) + "))");
	}

std::string WATGenerator::to_hex(uint8_t v) {
		char buf[4];
		snprintf(buf, sizeof(buf), "%02x", v);
		return buf;
	}

	// ============================================================
	// Emit full module
	// ============================================================

void WATGenerator::emit_module(std::ostringstream& out) {
		find_loop_ranges();

		out << "(module\n";
		out << "\n";
		out << "  ;; Host functions\n";

		// Imports
		emit_imports(out);

		// Memory
		size_t mem_pages = std::max((size_t)1024, (code_.size() * 128 + 65535) / 65536);
		out << "  (memory " << mem_pages << ")\n\n";

		// Execute function
		out << "  (func $execute (result i64)\n";

		// Body via relooper
		label_count_ = 0;
		auto result = emit_structured(0, code_.size());
		for(const auto& line : result.lines) {
			out << line << "\n";
		}

		out << "    (return (call $read_mem (i32.const 0)))\n";
		out << "  )\n\n";
		out << "  (export \"execute\" (func $execute))\n";
		out << ")\n";
	}

void WATGenerator::emit_imports(std::ostringstream& out) {
		// Gas + memory
		out << "  (import \"env\" \"use_gas\" (func $use_gas (param i32)))\n";
		out << "  (import \"env\" \"read_mem\" (func $read_mem (param i32) (result i64)))\n";
		out << "  (import \"env\" \"write_mem\" (func $write_mem (param i32 i64)))\n";
		out << "  (import \"env\" \"erase_mem\" (func $erase_mem (param i32)))\n";
		out << "  (import \"env\" \"op_copy\" (func $op_copy (param i32 i32)))\n";
		out << "  (import \"env\" \"op_clone\" (func $op_clone (param i32 i32)))\n";
		out << "  (import \"env\" \"op_call\" (func $op_call (param i32 i32)))\n";

		// uint256 arithmetic
		const char* u256_3[] = {
			"u256_add", "u256_sub", "u256_mul", "u256_div", "u256_mod",
			"u256_lt", "u256_gt", "u256_lte", "u256_gte",
			"u256_eq", "u256_neq", "u256_min", "u256_max",
			"u256_xor", "u256_and", "u256_or",
			"u256_lxor", "u256_land", "u256_lor",
			"u256_shl", "u256_shr"
		};
		for(const auto& name : u256_3) {
			out << "  (import \"env\" \"" << name << "\" (func $" << name << " (param i32 i32 i32)))\n";
		}

		// 2-param uint256
		out << "  (import \"env\" \"u256_not\" (func $u256_not (param i32 i32)))\n";
		out << "  (import \"env\" \"u256_lnot\" (func $u256_lnot (param i32 i32)))\n";

		// Container/type/crypto/blockchain
		out << "  (import \"env\" \"op_type\" (func $op_type (param i32 i32)))\n";
		out << "  (import \"env\" \"op_size\" (func $op_size (param i32 i32)))\n";
		out << "  (import \"env\" \"op_get_u256\" (func $op_get_u256 (param i32 i32 i32 i32)))\n";
		out << "  (import \"env\" \"op_set_u256\" (func $op_set_u256 (param i32 i32 i32)))\n";
		out << "  (import \"env\" \"op_erase\" (func $op_erase (param i32 i32)))\n";
		out << "  (import \"env\" \"op_push_back\" (func $op_push_back (param i32 i32)))\n";
		out << "  (import \"env\" \"op_pop_back\" (func $op_pop_back (param i32 i32)))\n";
		out << "  (import \"env\" \"op_conv\" (func $op_conv (param i32 i32 i32 i32)))\n";
		out << "  (import \"env\" \"op_concat\" (func $op_concat (param i32 i32 i32)))\n";
		out << "  (import \"env\" \"op_memcpy\" (func $op_memcpy (param i32 i32 i64 i64)))\n";
		out << "  (import \"env\" \"op_sha256\" (func $op_sha256 (param i32 i32)))\n";
		out << "  (import \"env\" \"op_verify\" (func $op_verify (param i32 i32 i32 i32)))\n";
		out << "  (import \"env\" \"op_log\" (func $op_log (param i64 i32)))\n";
		out << "  (import \"env\" \"op_send\" (func $op_send (param i32 i32 i32 i32)))\n";
		out << "  (import \"env\" \"op_mint\" (func $op_mint (param i32 i32 i32)))\n";
		out << "  (import \"env\" \"op_event\" (func $op_event (param i32 i32)))\n";
		out << "  (import \"env\" \"op_fail\" (func $op_fail (param i32 i64)))\n";
		out << "  (import \"env\" \"op_rcall\" (func $op_rcall (param i32 i32 i64 i64)))\n";
		out << "  (import \"env\" \"op_cread\" (func $op_cread (param i32 i32 i32)))\n";
		out << "  (import \"env\" \"op_balance\" (func $op_balance (param i32 i32)))\n";
		out << "\n";
}

// ============================================================
// Explicit method definitions (prevents -O3 from eliminating them)

WATGenerator::WATGenerator(const std::vector<instr_t>& code) : code_(code) {}

std::string WATGenerator::generate() {
	std::ostringstream out;
	emit_module(out);
	return out.str();
}

} // vm
} // mmx
