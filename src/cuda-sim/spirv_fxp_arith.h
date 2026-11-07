#pragma once
#include "spirv_fixed_point.h"

// CPU fixed-point arithmetic used by the PTX call runtime.
// Extracted from spirv_device_runtime.cc without changing the formulas.
// These routines are the research baseline. They are not an INT8 quantization
// model, and sub/div are not implemented here.

namespace spirv {
namespace _detail {
namespace fxp {

void binary_add(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp);

void sign_padding(unsigned op_bytes, FxpBinaryFuncOpTy& op);

void binary_mul(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp);

} // namespace fxp
} // namespace _detail
} // namespace spirv
