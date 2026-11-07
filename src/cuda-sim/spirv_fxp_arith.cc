#include "spirv_fxp_arith.h"

#include <cassert>
#include <cmath>
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

void binary_mul(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp){

  ret_exp = lhs_exp;

  float converter = ::pow(2.0f, (float)ret_exp);//fix mul only need to do one conversion

  unsigned raw_width = (lhs_size + rhs_size) << 3;
  unsigned ret_width = (lhs_size == rhs_size && raw_width <= 32)?
                        raw_width : 32;
  assert(ret_width == (ret_size << 3) &&
         "Return size mismatch");
  sign_padding(lhs_size, lhs_op);
  sign_padding(rhs_size, rhs_op);

  switch(ret_width){
    default:
      assert(false &&
             "Only support 8/16/32 bits for now");
    case 8:{
      int8_t lhs_val = static_cast<int16_t> (lhs_op.Single),
              rhs_val = static_cast<int16_t> (rhs_op.Single);
      int8_t result = lhs_val * rhs_val;
      result = result * converter;

      ret_val.Single = result;
      break;
    }

    case 16:{
      int16_t lhs_val = static_cast<int16_t> (lhs_op.Half),
              rhs_val = static_cast<int16_t> (rhs_op.Half);
      int16_t result = lhs_val * rhs_val;
      result = result * converter;

      ret_val.Half = result;

      break;
    }

    case 32:{
      int32_t lhs_val = static_cast<int32_t> (lhs_op.Full),
              rhs_val = static_cast<int32_t> (rhs_op.Full);
      int32_t result = lhs_val * rhs_val;
      result = result * converter;

      ret_val.Full = result;
      break;
    }
  }
}

} // namespace fxp
} // namespace _detail
} // namespace spirv
