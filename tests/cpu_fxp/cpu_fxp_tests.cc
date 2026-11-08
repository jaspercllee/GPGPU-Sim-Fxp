// Standalone CPU checks for the fixed-point baseline.
// Checks are ordinary if-statements, so -DNDEBUG / release builds still run them.
// This file does not link GPGPU-Sim, CUDA, or OpenCL.

#include "spirv_fixed_point.h"
#include "spirv_fxp_arith.h"
#include "../../experiments/mnist_mlp/fxp_requant.h"
#include "../../src/gpuwattch/fxp_power_policy.h"

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

void check_mul() {
  // Corrected contract: raw product and exponent sum.
  // The 2020 runtime instead kept lhs_exp and multiplied by pow(2, lhs_exp)
  // in float (7*8 at -2,-2 used to be exponent -2 and raw 14).
  // The int64 product is then narrowed. 32x32 keeps the low 32 bits, which
  // is what the 2020 gtest stores in a 4-byte return. That is not the paper's
  // unrestricted Wa+Wb storage.
  FxpBinaryFuncOpTy lhs = {7};
  FxpBinaryFuncOpTy rhs = {8};
  FxpBinaryFuncOpTy ret = {0};
  int exp = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, -2, 4, rhs, -2, 4, ret, exp);
  CHECK(exp == -4);
  CHECK(ret.Full == 56);

  lhs.Full = 7;
  rhs.Full = 8;
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, -2, 4, rhs, -3, 4, ret, exp);
  CHECK(exp == -5);
  CHECK(ret.Full == 56);

  lhs.Full = static_cast<uint32_t>(-7);
  rhs.Full = 8;
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, -2, 4, rhs, -2, 4, ret, exp);
  CHECK(exp == -4);
  CHECK(ret.Full == static_cast<uint32_t>(-56));

  lhs.Full = static_cast<uint32_t>(-7);
  rhs.Full = static_cast<uint32_t>(-8);
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, 3, 4, rhs, -1, 4, ret, exp);
  CHECK(exp == 2);
  CHECK(ret.Full == 56);

  lhs.Full = (1u << 20) - 1u;
  rhs.Full = (1u << 15) - 1u;
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, -2, 4, rhs, -3, 4, ret, exp);
  CHECK(exp == -5);
  CHECK(ret.Full == 4293885953u);

  lhs.Full = 0x80000000u;
  rhs.Full = static_cast<uint32_t>(-1);
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, 0, 4, rhs, 0, 4, ret, exp);
  CHECK(exp == 0);
  CHECK(ret.Full == 0x80000000u);

  lhs.Full = 0;
  rhs.Full = 0;
  lhs.Single = static_cast<uint8_t>(-128);
  rhs.Single = static_cast<uint8_t>(-128);
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(1, lhs, -1, 1, rhs, -1, 2, ret, exp);
  CHECK(exp == -2);
  CHECK(ret.Half == 16384);

  lhs.Full = 0;
  rhs.Full = 0;
  lhs.Half = 32767;
  rhs.Half = 32767;
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(2, lhs, 1, 2, rhs, 1, 4, ret, exp);
  CHECK(exp == 2);
  CHECK(ret.Full == 1073676289u);

  lhs.Full = 0;
  rhs.Full = 2;
  lhs.Single = static_cast<uint8_t>(-128);
  ret.Full = 0;
  spirv::_detail::fxp::binary_mul(1, lhs, -3, 4, rhs, -1, 4, ret, exp);
  CHECK(exp == -4);
  CHECK(ret.Full == static_cast<uint32_t>(-256));
}

void check_sub() {
  // min exponent, max width, value-preserving left shift, then wrap.
  FxpBinaryFuncOpTy lhs = {9};
  FxpBinaryFuncOpTy rhs = {4};
  FxpBinaryFuncOpTy ret = {0};
  int exp = 0;
  spirv::_detail::fxp::binary_sub(4, lhs, -2, 4, rhs, -2, 4, ret, exp);
  CHECK(exp == -2);
  CHECK(ret.Full == 5);

  lhs.Full = 8;
  rhs.Full = 7;
  ret.Full = 0;
  spirv::_detail::fxp::binary_sub(4, lhs, -3, 4, rhs, -4, 4, ret, exp);
  CHECK(exp == -4);
  CHECK(ret.Full == 9);

  lhs.Full = static_cast<uint32_t>(-3);
  rhs.Full = 5;
  ret.Full = 0;
  spirv::_detail::fxp::binary_sub(4, lhs, -2, 4, rhs, -2, 4, ret, exp);
  CHECK(exp == -2);
  CHECK(ret.Full == static_cast<uint32_t>(-8));

  lhs.Full = 0;
  rhs.Full = 0;
  lhs.Single = 1;
  rhs.Single = 2;
  ret.Full = 0;
  spirv::_detail::fxp::binary_sub(1, lhs, 0, 1, rhs, 0, 1, ret, exp);
  CHECK(exp == 0);
  CHECK(ret.Single == 0xFFu);

  lhs.Full = 0x80000000u;
  rhs.Full = 1;
  ret.Full = 0;
  spirv::_detail::fxp::binary_sub(4, lhs, 0, 4, rhs, 0, 4, ret, exp);
  CHECK(exp == 0);
  CHECK(ret.Full == 0x7FFFFFFFu);

  lhs.Full = 1;
  rhs.Full = 0;
  ret.Full = 0;
  spirv::_detail::fxp::binary_sub(4, lhs, 0, 4, rhs, -80, 4, ret, exp);
  CHECK(exp == -80);
  CHECK(ret.Full == 0);

  lhs.Full = 0;
  rhs.Full = 3;
  lhs.Single = static_cast<uint8_t>(-2);
  ret.Full = 0;
  spirv::_detail::fxp::binary_sub(1, lhs, -1, 4, rhs, -2, 4, ret, exp);
  CHECK(exp == -2);
  CHECK(ret.Full == static_cast<uint32_t>(-7));
}

void check_requant_mac() {
  // 1.5 and 2.0 at exponent -2. Product exponent is -4 (raw 48 = 3.0).
  // Requantize back to -2 before the add. Do not add the raw -4 value to a -2 accumulator.
  CHECK(fxp_quantize_f64(1.5, -2, 32) == 6);
  CHECK(fxp_quantize_f64(2.0, -2, 32) == 8);
  FxpBinaryFuncOpTy lhs = {6};
  FxpBinaryFuncOpTy rhs = {8};
  FxpBinaryFuncOpTy prod = {0};
  int exp = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, -2, 4, rhs, -2, 4, prod, exp);
  CHECK(exp == -4);
  CHECK(prod.Full == 48);
  CHECK(fxp_requantize(int64_t(prod.Full), -4, -2, 32) == 12);
  // 5 * 2^{-2} requantized to 2^{-1} is 2.5, half away from zero -> 3.
  CHECK(fxp_requantize(5, -2, -1, 32) == 3);
  CHECK(fxp_requantize(-5, -2, -1, 32) == -3);
  FxpBinaryFuncOpTy acc = {12};
  FxpBinaryFuncOpTy extra = {4};  // 1.0 at exponent -2
  FxpBinaryFuncOpTy sum = {0};
  int sum_exp = 0;
  spirv::_detail::fxp::binary_add(4, acc, -2, 4, extra, -2, 4, sum, sum_exp);
  CHECK(sum_exp == -2);
  CHECK(sum.Full == 16);
  // Two-tap dot: 1.5*2.0 + 0.5*4.0 = 5.0, stored as 20 at exponent -2.
  auto tap = [](uint32_t a, uint32_t b) -> int32_t {
    FxpBinaryFuncOpTy lhs = {a};
    FxpBinaryFuncOpTy rhs = {b};
    FxpBinaryFuncOpTy prod = {0};
    int exp = 0;
    spirv::_detail::fxp::binary_mul(4, lhs, -2, 4, rhs, -2, 4, prod, exp);
    CHECK(exp == -4);
    return int32_t(fxp_requantize(int64_t(prod.Full), -4, -2, 32));
  };
  const int32_t t0 = tap(6, 8);
  const int32_t t1 = tap(2, 16);
  CHECK(t0 == 12);
  CHECK(t1 == 8);
  FxpBinaryFuncOpTy d0 = {uint32_t(t0)};
  FxpBinaryFuncOpTy d1 = {uint32_t(t1)};
  FxpBinaryFuncOpTy dot = {0};
  int dot_exp = 0;
  spirv::_detail::fxp::binary_add(4, d0, -2, 4, d1, -2, 4, dot, dot_exp);
  CHECK(dot_exp == -2);
  CHECK(dot.Full == 20);
  CHECK(fxp_sat_signed(200, 8) == 127);
  CHECK(fxp_sat_signed(-200, 8) == -128);

  // CNN contract: both factors are exponent -12, so the product exponent is
  // -24 and is not returned by the simulator. The PTX right-shifts 12 bits,
  // rounding half away from zero, including negative magnitudes and INT_MIN.
  spirv::FxpBinaryFunction cnn_mul("_Z7fxp_mul12fxp_32_12_3_12fxp_32_12_3_");
  CHECK(cnn_mul.Valid);
  CHECK(cnn_mul.OpName == "mul");
  CHECK(cnn_mul.LhsTotalBits == 32u);
  CHECK(cnn_mul.RhsTotalBits == 32u);
  CHECK(cnn_mul.LhsExponent == -12);
  CHECK(cnn_mul.RhsExponent == -12);
  auto shr12 = [](int32_t q) -> int32_t {
    if (q == INT32_MIN) return -524288;
    const uint32_t mag = static_cast<uint32_t>(q < 0 ? -q : q);
    uint32_t hi = mag >> 12;
    if ((mag & 4095u) >= 2048u) hi += 1u;
    const int32_t out = static_cast<int32_t>(hi);
    return q < 0 ? -out : out;
  };
  int shr_mismatch = 0;
  for (int q = -100000; q <= 100000; ++q) {
    if (shr12(static_cast<int32_t>(q)) !=
        static_cast<int32_t>(fxp_requantize(q, -24, -12, 32)))
      ++shr_mismatch;
  }
  const int32_t edges[] = {
      0, 1, -1, 2047, 2048, -2047, -2048, 4095, 4096, -4095, -4096,
      5000, -5000, INT32_MAX, INT32_MIN};
  for (unsigned i = 0; i < sizeof(edges) / sizeof(edges[0]); ++i) {
    if (shr12(edges[i]) !=
        static_cast<int32_t>(fxp_requantize(edges[i], -24, -12, 32)))
      ++shr_mismatch;
  }
  CHECK(shr_mismatch == 0);
  CHECK(shr12(-2048) == -1);
  CHECK(shr12(-2047) == 0);
  CHECK(shr12(2048) == 1);
  CHECK(shr12(INT32_MIN) == -524288);
  auto mul_rq = [](int32_t a, int32_t b) -> int32_t {
    FxpBinaryFuncOpTy lhs = {static_cast<uint32_t>(a)};
    FxpBinaryFuncOpTy rhs = {static_cast<uint32_t>(b)};
    FxpBinaryFuncOpTy prod = {0};
    int exp = 0;
    spirv::_detail::fxp::binary_mul(4, lhs, -12, 4, rhs, -12, 4, prod, exp);
    CHECK(exp == -24);
    return static_cast<int32_t>(fxp_requantize(static_cast<int32_t>(prod.Full), -24, -12, 32));
  };
  CHECK(mul_rq(-100, 50) == shr12(-5000));
  CHECK(mul_rq(100, -50) == shr12(-5000));
  CHECK(mul_rq(-100, -50) == shr12(5000));
  CHECK(mul_rq(INT32_MIN, 1) == -524288);
  CHECK(mul_rq(-2048, 1) == -1);
  CHECK(mul_rq(2047, 1) == 0);
  // One 3x3 accumulator step: bias, then mul/requant/add, then negative ReLU.
  int32_t acc32 = 7;
  const int32_t taps_a[] = {3000, -100, 2048};
  const int32_t taps_b[] = {-3, 50, 1};
  for (int i = 0; i < 3; ++i) {
    const int32_t aligned = mul_rq(taps_a[i], taps_b[i]);
    FxpBinaryFuncOpTy lhs = {static_cast<uint32_t>(acc32)};
    FxpBinaryFuncOpTy rhs = {static_cast<uint32_t>(aligned)};
    FxpBinaryFuncOpTy sum = {0};
    int sum_exp = 0;
    spirv::_detail::fxp::binary_add(4, lhs, -12, 4, rhs, -12, 4, sum, sum_exp);
    CHECK(sum_exp == -12);
    acc32 = static_cast<int32_t>(sum.Full);
  }
  if (acc32 < 0) acc32 = 0;
  int32_t indep = 7;
  for (int i = 0; i < 3; ++i) {
    const int64_t wide = int64_t(taps_a[i]) * int64_t(taps_b[i]);
    indep += shr12(static_cast<int32_t>(wide));
  }
  if (indep < 0) indep = 0;
  CHECK(acc32 == indep);
}

void check_fxp_power_policy() {
  // Absent XML uses the integer coefficient. An explicit XML value is kept.
  // Neither path multiplies by a bit width.
  CHECK(fxp_activity_coefficient(0, 99.0, 4.57) == 4.57);
  CHECK(fxp_activity_coefficient(1, 1.0, 4.57) == 1.0);
  CHECK(fxp_activity_coefficient(1, 4.57, 4.57) == 4.57);
}

void check_div() {
  // Engineering policy. The paper does not specify division.
  // Trunc toward zero. Divisor 0 yields raw 0. No saturation.
  FxpBinaryFuncOpTy lhs = {7};
  FxpBinaryFuncOpTy rhs = {2};
  FxpBinaryFuncOpTy ret = {0};
  int exp = 0;
  spirv::_detail::fxp::binary_div(4, lhs, -2, 4, rhs, -2, 4, ret, exp);
  CHECK(exp == 0);
  CHECK(ret.Full == 3);

  lhs.Full = static_cast<uint32_t>(-7);
  rhs.Full = 2;
  ret.Full = 0;
  spirv::_detail::fxp::binary_div(4, lhs, -2, 4, rhs, -2, 4, ret, exp);
  CHECK(ret.Full == static_cast<uint32_t>(-3));

  lhs.Full = 7;
  rhs.Full = static_cast<uint32_t>(-2);
  ret.Full = 0;
  spirv::_detail::fxp::binary_div(4, lhs, 0, 4, rhs, 0, 4, ret, exp);
  CHECK(ret.Full == static_cast<uint32_t>(-3));

  lhs.Full = static_cast<uint32_t>(-7);
  rhs.Full = static_cast<uint32_t>(-2);
  ret.Full = 0;
  spirv::_detail::fxp::binary_div(4, lhs, -2, 4, rhs, -4, 4, ret, exp);
  CHECK(exp == 2);
  CHECK(ret.Full == 3);

  lhs.Full = 8;
  rhs.Full = 2;
  ret.Full = 0;
  spirv::_detail::fxp::binary_div(4, lhs, -2, 4, rhs, -4, 4, ret, exp);
  CHECK(exp == 2);
  CHECK(ret.Full == 4);

  lhs.Full = 5;
  rhs.Full = 0;
  ret.Full = 99;
  spirv::_detail::fxp::binary_div(4, lhs, -3, 4, rhs, -1, 4, ret, exp);
  CHECK(exp == -2);
  CHECK(ret.Full == 0);

  lhs.Full = 0x80000000u;
  rhs.Full = static_cast<uint32_t>(-1);
  ret.Full = 0;
  spirv::_detail::fxp::binary_div(4, lhs, 0, 4, rhs, 0, 4, ret, exp);
  CHECK(exp == 0);
  CHECK(ret.Full == 0x80000000u);

  lhs.Full = 0;
  rhs.Full = 0;
  lhs.Half = 0x8000u;
  rhs.Half = 0xFFFFu;
  ret.Full = 0;
  spirv::_detail::fxp::binary_div(2, lhs, 1, 2, rhs, 4, 2, ret, exp);
  CHECK(exp == -3);
  CHECK(ret.Half == 0x8000u);
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
  check_mul();
  check_sub();
  check_div();
  check_requant_mac();
  check_fxp_power_policy();
  std::printf("cpu_fxp_tests: %d passed, %d failed\n", g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}
