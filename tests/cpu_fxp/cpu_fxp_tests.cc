// Standalone CPU checks for the fixed-point baseline.
// Checks are ordinary if-statements, so -DNDEBUG / release builds still run them.
// This file does not link GPGPU-Sim, CUDA, or OpenCL.

#include "spirv_fixed_point.h"
#include "spirv_fxp_arith.h"

#include <cstdio>
#include <cstdint>
#include <string>

namespace {

int g_failed = 0;
int g_passed = 0;

void check(bool ok, const char* expr, const char* file, int line) {
  if (ok) {
    ++g_passed;
    return;
  }
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
  ++g_failed;
}

#define CHECK(expr) check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

using spirv::_detail::fxp::FxpBinaryFuncOpTy;

void check_parser() {
  CHECK(!spirv::IsFxpBinaryFunction("abcdefg"));
  CHECK(spirv::IsFxpBinaryFunction("_Z7fxp_div12fxp_32_28_3_12fxp_32_28_3_"));
  CHECK(spirv::IsFxpBinaryFunction("_Z7fxp_mul12fxp_32_28_3_12fxp_32_28_3_"));
  // Legacy SPIR-V style mangling is intentionally not accepted.
  CHECK(!spirv::IsFxpBinaryFunction(
      "_Z14__spirv_FxpAddPU3AS425__spirv_FixedPoint__8_3_3PU3AS426__spirv_FixedPoint__16_3_3"));

  spirv::FxpBinaryFunction add("_Z7fxp_add12fxp_8_3_3_12fxp_16_3_3_");
  CHECK(add.Valid);
  CHECK(add.OpName == "add");
  CHECK(add.LhsTotalBits == 8);
  CHECK(add.LhsExponent == -3);
  CHECK(add.RhsTotalBits == 16);
  CHECK(add.RhsExponent == -3);

  spirv::FxpBinaryFunction sub("_Z7fxp_sub12fxp_8_3_3_12fxp_16_4_3_");
  CHECK(sub.Valid);
  CHECK(sub.OpName == "sub");
  CHECK(sub.LhsTotalBits == 8);
  CHECK(sub.LhsExponent == -3);
  CHECK(sub.RhsTotalBits == 16);
  CHECK(sub.RhsExponent == -4);

  spirv::FxpBinaryFunction mul("_Z7fxp_mul12fxp_8_3_3_12fxp_8_5_1_");
  CHECK(mul.Valid);
  CHECK(mul.OpName == "mul");
  CHECK(mul.LhsExponent == -3);
  CHECK(mul.RhsExponent == 5);

  spirv::FxpBinaryFunction divop("_Z7fxp_div12fxp_32_33_2_12fxp_8_5_1_");
  CHECK(divop.Valid);
  CHECK(divop.OpName == "div");
  CHECK(divop.LhsTotalBits == 32);
  CHECK(divop.LhsExponent == -33);
  CHECK(divop.RhsTotalBits == 8);
  CHECK(divop.RhsExponent == 5);

  CHECK(!spirv::IsFxpConvertFunction("abcde"));
  CHECK(!spirv::IsFxpConvertFunction(""));
  CHECK(spirv::IsFxpConvertFunction("_Z24convert_fxp_32_33_3__rtei"));
  CHECK(spirv::IsFxpConvertFunction("_Z24convert_fxp_16_8_2__rtef"));
  CHECK(spirv::IsFxpConvertFunction("_Z24convert_fxp_8_7_2__rtnfxp_32_33_3_"));
  CHECK(spirv::IsFxpConvertFunction("_Z17convert_float_rtefxp_32_13_3_"));

  spirv::FxpConvertFunction to_fxp("_Z24convert_fxp_32_33_3__rtei");
  CHECK(to_fxp.Valid);
  CHECK(to_fxp.SrcTy.get() != nullptr);
  CHECK(spirv::dyn_cast<spirv::IntType>(to_fxp.SrcTy.get()) != nullptr);
  spirv::FxpType* dest = spirv::dyn_cast<spirv::FxpType>(to_fxp.DestTy.get());
  CHECK(dest != nullptr);
  if (dest) {
    CHECK(dest->Width == 32);
    CHECK(dest->Exponent == -33);
  }

  spirv::utils::CXXABIParser::result_type parsed;
  CHECK(spirv::IsFxpMathFunction("_Z3exp10fxp_8_5_1_", &parsed));
  CHECK(parsed.size() == 2);
  CHECK(parsed.size() == 2 && parsed.at(0) == "exp");
  CHECK(parsed.size() == 2 && parsed.at(1) == "fxp_8_5_1_");

  spirv::utils::CXXABIParser::result_type exp10;
  CHECK(spirv::IsFxpMathFunction("_Z5exp1010fxp_8_5_1_", &exp10));
  CHECK(exp10.size() == 2 && exp10.at(0) == "exp10");

  std::unique_ptr<spirv::FxpMathFunction> log10 =
      spirv::FxpMathFunction::Create("_Z5log1010fxp_8_5_3_");
  CHECK(log10.get() != nullptr);
  if (log10) {
    CHECK(log10->Valid);
    CHECK(log10->Name == "log10");
    CHECK(log10->ArgTypes.size() == 1);
    spirv::FxpType* arg = spirv::dyn_cast<spirv::FxpType>(log10->ArgTypes.at(0));
    CHECK(arg != nullptr);
    if (arg) {
      CHECK(arg->Width == 8);
      CHECK(arg->Exponent == -5);
      CHECK(arg->Metadata == 3);
      CHECK(arg->Signed);
    }
  }
}

void check_add_and_padding() {
  // Same cases as tests/spirv/fixed_point_arithmetic_impl.cc. Those add
  // expectations match the current implementation.
  FxpBinaryFuncOpTy lhs = {9};
  FxpBinaryFuncOpTy rhs = {4};
  FxpBinaryFuncOpTy ret = {0};
  int exp = 0;
  spirv::_detail::fxp::binary_add(4, lhs, -2, 4, rhs, -2, 4, ret, exp);
  CHECK(exp == -2);
  CHECK(ret.Full == 13);

  lhs.Full = 8;
  rhs.Full = 7;
  ret.Full = 0;
  spirv::_detail::fxp::binary_add(4, lhs, -3, 4, rhs, -4, 4, ret, exp);
  CHECK(exp == -4);
  CHECK(ret.Full == 23);

  lhs.Full = 9;
  rhs.Full = 4;
  ret.Full = 0;
  spirv::_detail::fxp::binary_add(2, lhs, -5, 1, rhs, -5, 2, ret, exp);
  CHECK(exp == -5);
  CHECK(ret.Half == 13);

  FxpBinaryFuncOpTy u32 = {7};
  spirv::_detail::fxp::sign_padding(4, u32);
  CHECK(u32.Full == 7);

  FxpBinaryFuncOpTy u8 = {(1u << 30) - 1u};
  u8.Single = 7;
  spirv::_detail::fxp::sign_padding(1, u8);
  CHECK(u8.Half == 7);

  FxpBinaryFuncOpTy s8 = {0};
  s8.Single = 135;
  spirv::_detail::fxp::sign_padding(1, s8);
  CHECK(static_cast<int16_t>(s8.Half) == -121);
}

void check_mul_baseline() {
  // tests/spirv/fixed_point_arithmetic_impl.cc expects a pure integer product
  // and an exponent of lhs_exp + rhs_exp (for example -4 and raw 56).
  // spirv_fxp_arith.cc instead keeps lhs_exp and scales the wrapped product by
  // pow(2, lhs_exp) in float. This test locks the implemented behavior.
  // It does not change that behavior and it does not redefine the 2020 gtest.
  FxpBinaryFuncOpTy lhs = {7};
  FxpBinaryFuncOpTy rhs = {8};
  FxpBinaryFuncOpTy ret = {0};
  int exp = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, -2, 4, rhs, -2, 4, ret, exp);
  CHECK(exp == -2);
  CHECK(ret.Full == 14);

  lhs.Full = (1u << 20) - 1u;
  rhs.Full = (1u << 15) - 1u;
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, -2, 4, rhs, -3, 4, ret, exp);
  CHECK(exp == -2);
  CHECK(ret.Full == 4294696961u);
}

} // namespace

int main() {
  std::printf("cpu_fxp_tests: release-safe checks, NDEBUG=%d\n",
#ifdef NDEBUG
              1
#else
              0
#endif
  );
  check_parser();
  check_add_and_padding();
  check_mul_baseline();
  std::printf("cpu_fxp_tests: %d passed, %d failed\n", g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}
