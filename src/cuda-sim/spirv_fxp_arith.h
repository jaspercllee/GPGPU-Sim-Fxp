#pragma once
#include "spirv_fixed_point.h"

// CPU fixed-point arithmetic used by the PTX call runtime.
// binary_add and sign_padding keep the 2020 formulas.
// binary_mul returns a raw integer product and the sum of the exponents.
// binary_sub aligns to the smaller exponent, then subtracts.
// binary_div is an engineering policy: the paper does not specify division.
// None of these is an INT8 quantization model.

namespace spirv {
namespace _detail {
namespace fxp {

void binary_add(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp);

void sign_padding(unsigned op_bytes, FxpBinaryFuncOpTy& op);

void binary_sub(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp);

void binary_mul(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp);

void binary_div(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp);

} // namespace fxp
} // namespace _detail
} // namespace spirv
