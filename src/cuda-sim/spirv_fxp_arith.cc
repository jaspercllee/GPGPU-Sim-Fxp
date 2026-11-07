#include "spirv_fxp_arith.h"

#include <cassert>
#include <cmath>
#include <climits>
#include <cstdint>

namespace spirv {
namespace _detail {
namespace fxp {

static inline unsigned s2u_abs(int S){
  return (S >= 0)? (unsigned)S : (unsigned)-S;
}

#define FXP_EXTRACT(width, fxp_val, action) \
  switch(width << 3){ \
  case 8:{ \
    uint8_t val = fxp_val.Single; \
    action \
    break; \
  } \
  case 16:{ \
    uint16_t val = fxp_val.Half; \
    action \
    break; \
  } \
  case 32:{ \
    uint32_t val = fxp_val.Full; \
    action \
    break; \
  } \
  default: \
    assert(false && "Invalid fxp width"); \
  }

template<typename T>
static inline T shift_fxp(T storage, int bias){
  if(bias == 0) return storage;

  if(bias > 0){
    return (storage >> s2u_abs(bias));
  }else{
    return (storage << s2u_abs(bias));
  }
}

template<typename T>
static inline T do_binary_add(T lhs_val, int lhs_bias,
                              T rhs_val, int rhs_bias){
  lhs_val = shift_fxp(lhs_val, lhs_bias);
  rhs_val = shift_fxp(rhs_val, rhs_bias);

  return lhs_val + rhs_val;
}

void binary_add(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp){
  /*
   * Smaller storage bits would be promoted
   */
  unsigned size = (rhs_size > lhs_size)? rhs_size : lhs_size;
  assert(size == ret_size &&
         "Return size mismatch");

  /*
   * Take the smaller exponent
   */
  int exponent = (rhs_exp < lhs_exp)? rhs_exp : lhs_exp;
  ret_exp = exponent;
  int lhs_bias = exponent - lhs_exp,
      rhs_bias = exponent - rhs_exp;

  float converter = ::pow(2.0f, (float)ret_exp);

  switch(size << 3){
    default:
      assert(false &&
             "Only support 8/16/32 bits for now");
    case 8:{
      uint8_t lhs_val = lhs_op.Single,
              rhs_val = rhs_op.Single;
      uint8_t result = do_binary_add(lhs_val, lhs_bias,
                                     rhs_val, rhs_bias);
      ret_val.Single = result;
      break;
    }

    case 16:{
      uint16_t lhs_val = lhs_op.Half,
               rhs_val = rhs_op.Half;
      uint16_t result = do_binary_add(lhs_val, lhs_bias,
                                      rhs_val, rhs_bias);
      ret_val.Half = result;
      break;
    }

    case 32:{
      int32_t lhs_val = static_cast<int32_t> (lhs_op.Full),
              rhs_val = static_cast<int32_t> (rhs_op.Full);
      int32_t result = do_binary_add(lhs_val, lhs_bias,
                                     rhs_val, rhs_bias);
      ret_val.Full = result;
      break;
    }
  }
  (void)converter;
}

void sign_padding(unsigned op_bytes, FxpBinaryFuncOpTy& op) {
  FXP_EXTRACT(op_bytes, op, {
    using val_t = decltype(val);

    val_t scratch = val;
    val_t mask = 1;
    unsigned op_width = (op_bytes << 3);
    scratch >>= (op_width - 1);
    bool isSigned = scratch & mask;

    uint32_t op_mask = 0;
    op_mask = ~op_mask;
    // It is undefined if the left shift amount
    // is larger or equal to the LHS bit width
    if(op_width >= 32) op_mask = 0;
    else op_mask <<= op_width;

    if(isSigned) op.Full |= op_mask;
    else op.Full &= ~op_mask;
  })
}

// Low 32 bits of a two's-complement sum, without signed-overflow UB and
// without an out-of-range conversion to int. Exponent sums outside
// [INT_MIN, INT_MAX] are not a supported contract; the bits are still defined.
static int narrow_exp(long long sum) {
  const uint32_t low = static_cast<uint32_t>(static_cast<unsigned long long>(sum));
  if (low <= 2147483647u)
    return static_cast<int>(low);
  return static_cast<int>(low - 2147483648u) + INT_MIN;
}

static int defined_exp_sum(int lhs_exp, int rhs_exp) {
  return narrow_exp(static_cast<long long>(lhs_exp) +
                    static_cast<long long>(rhs_exp));
}

static int defined_exp_diff(int lhs_exp, int rhs_exp) {
  return narrow_exp(static_cast<long long>(lhs_exp) -
                    static_cast<long long>(rhs_exp));
}

static void store_low_bits(FxpBinaryFuncOpTy& dst, unsigned ret_width_bits,
                           uint64_t bits) {
  switch (ret_width_bits) {
    case 8:
      dst.Single = static_cast<uint8_t>(bits);
      break;
    case 16:
      dst.Half = static_cast<uint16_t>(bits);
      break;
    case 32:
      dst.Full = static_cast<uint32_t>(bits);
      break;
    default:
      assert(false && "Only support 8/16/32 bit results for now");
  }
}

static void store_wrapped_product(FxpBinaryFuncOpTy& dst, unsigned ret_width_bits,
                                  int64_t product) {
  store_low_bits(dst, ret_width_bits, static_cast<uint64_t>(product));
}

static unsigned wider_bytes(unsigned lhs_size, unsigned rhs_size) {
  return (rhs_size > lhs_size) ? rhs_size : lhs_size;
}

// Left-align a sign-extended raw value toward a smaller exponent.
// shift >= 64 does not fit in the 64-bit working register: the aligned
// value is defined as 0. That limit is not a paper rule.
static uint64_t align_toward_smaller_exp(int32_t raw, int op_exp, int result_exp) {
  const long long shift = static_cast<long long>(op_exp) -
                          static_cast<long long>(result_exp);
  uint64_t bits = static_cast<uint64_t>(static_cast<int64_t>(raw));
  if (shift <= 0)
    return bits;
  if (shift >= 64)
    return 0;
  return bits << static_cast<unsigned>(shift);
}

void binary_mul(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp){
  // Corrected contract: raw integer product, result exponent is the sum.
  // The 2020 routine kept lhs_exp and multiplied the product by pow(2, lhs_exp)
  // in float. That rescale is gone. 7 * 8 with exponents -2 and -2 is raw 56
  // and exponent -4.
  ret_exp = defined_exp_sum(lhs_exp, rhs_exp);

  // The int64 product is the full intermediate of the sign-extended raw
  // values. Paper Algorithm 2 also asks for a stored width of Wa+Wb. This
  // helper still uses the historical return-width rule, because the storage
  // union and the 2020 gtest return 4 bytes and expect 32-bit wrap:
  // equal byte sizes whose summed bit widths fit in 32 keep that sum
  // (8x8 -> 16, 16x16 -> 32); every other pair, including 32x32, is narrowed
  // to the low 32 bits. The formula never selects 8.
  unsigned raw_width = (lhs_size + rhs_size) << 3;
  unsigned ret_width = (lhs_size == rhs_size && raw_width <= 32)?
                        raw_width : 32;
  assert(ret_width == (ret_size << 3) &&
         "Return size mismatch");
  sign_padding(lhs_size, lhs_op);
  sign_padding(rhs_size, rhs_op);

  // sign_padding writes a 32-bit two's-complement value. Multiplying those
  // int32 values in int64 is defined for every pair, including INT_MIN * -1.
  // The stored result is the low ret_width bits. There is no saturation.
  const int64_t lhs_val = static_cast<int32_t>(lhs_op.Full);
  const int64_t rhs_val = static_cast<int32_t>(rhs_op.Full);
  store_wrapped_product(ret_val, ret_width, lhs_val * rhs_val);
}

void binary_sub(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp){
  // Paper Algorithm 2: exponent is the minimum, stored width is the maximum.
  // The pseudocode writes LHS-RHS without showing a shift. Aligning the
  // larger exponent by a left shift is required to keep N*2^E, and it is
  // recorded here as that derived rule. There is no saturation. Bits that
  // do not fit in the stored width wrap. binary_add is a separate routine
  // and is not changed by this function.
  const int result_exp = (rhs_exp < lhs_exp) ? rhs_exp : lhs_exp;
  ret_exp = result_exp;
  const unsigned size = wider_bytes(lhs_size, rhs_size);
  assert(size == ret_size && "Return size mismatch");
  assert((size == 1 || size == 2 || size == 4) &&
         "Only support 8/16/32 bits for now");

  sign_padding(lhs_size, lhs_op);
  sign_padding(rhs_size, rhs_op);
  const uint64_t lhs_bits = align_toward_smaller_exp(
      static_cast<int32_t>(lhs_op.Full), lhs_exp, result_exp);
  const uint64_t rhs_bits = align_toward_smaller_exp(
      static_cast<int32_t>(rhs_op.Full), rhs_exp, result_exp);
  store_low_bits(ret_val, size << 3, lhs_bits - rhs_bits);
}

void binary_div(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp){
  // Not specified by the paper. Chosen policy:
  //   exponent = lhs_exp - rhs_exp
  //   raw = trunc-toward-zero quotient of the sign-extended integers
  //   stored width = max operand width, then low bits if the quotient
  //   does not fit (no saturation)
  //   divisor 0 stores raw 0 and still records the exponent difference
  // Conversion rounding modes are not used.
  ret_exp = defined_exp_diff(lhs_exp, rhs_exp);
  const unsigned size = wider_bytes(lhs_size, rhs_size);
  assert(size == ret_size && "Return size mismatch");
  assert((size == 1 || size == 2 || size == 4) &&
         "Only support 8/16/32 bits for now");

  sign_padding(lhs_size, lhs_op);
  sign_padding(rhs_size, rhs_op);
  const int64_t lhs_val = static_cast<int32_t>(lhs_op.Full);
  const int64_t rhs_val = static_cast<int32_t>(rhs_op.Full);
  int64_t quotient = 0;
  if (rhs_val != 0)
    quotient = lhs_val / rhs_val;
  store_wrapped_product(ret_val, size << 3, quotient);
}

} // namespace fxp
} // namespace _detail
} // namespace spirv
