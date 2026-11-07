#pragma once
// Conversion used by the MNIST MLP driver. This is not binary_add/binary_mul.
// q_to * 2^exp_to approximates q_from * 2^exp_from.
// Rounding is half away from zero. The stored result then saturates to a
// signed 8/16/32-bit range. Saturation here is only this conversion step.

#include <cstdint>
#include <cstdlib>

inline int64_t fxp_sat_signed(int64_t v, int width_bits) {
  if (width_bits < 2 || width_bits > 32)
    std::abort();
  const int64_t maxv = (int64_t(1) << (width_bits - 1)) - 1;
  const int64_t minv = -(int64_t(1) << (width_bits - 1));
  if (v > maxv) return maxv;
  if (v < minv) return minv;
  return v;
}

inline int64_t fxp_requantize(int64_t q, int exp_from, int exp_to, int width_bits) {
  const int sh = exp_from - exp_to;
  int64_t v = q;
  if (sh > 0) {
    if (sh >= 62) {
      v = 0;
    } else if (q > (INT64_C(0x3fffffffffffffff) >> (sh - 1)) ||
               q < -((INT64_C(0x3fffffffffffffff) >> (sh - 1)))) {
      v = (q < 0) ? INT64_MIN : INT64_MAX;
    } else {
      v = q * (int64_t(1) << sh);
    }
  } else if (sh < 0) {
    const int r = -sh;
    if (r >= 62) {
      v = 0;
    } else {
      const uint64_t mag = (q < 0)
        ? ((q == INT64_MIN) ? (uint64_t(1) << 63) : uint64_t(-q))
        : uint64_t(q);
      const uint64_t rounded = (mag + (uint64_t(1) << (r - 1))) >> r;
      v = (q < 0) ? -int64_t(rounded) : int64_t(rounded);
    }
  }
  return fxp_sat_signed(v, width_bits);
}

inline int32_t fxp_quantize_f64(double x, int exp, int width_bits) {
  // q ~= x * 2^{-exp}
  int scale_exp = -exp;
  double mag = x < 0 ? -x : x;
  double scaled = mag;
  if (scale_exp >= 0) {
    if (scale_exp < 62)
      scaled *= double(int64_t(1) << scale_exp);
    else
      scaled = 1e300;
  } else if (-scale_exp < 62) {
    scaled /= double(int64_t(1) << -scale_exp);
  } else {
    scaled = 0;
  }
  int64_t q = int64_t(scaled + 0.5);
  if (x < 0) q = -q;
  return int32_t(fxp_sat_signed(q, width_bits));
}
