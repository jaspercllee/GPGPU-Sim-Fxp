#include <gtest/gtest.h>
#include "spirv_fixed_point.h"

namespace spirv {
namespace _detail{
namespace fxp{

extern
void binary_add(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp);

extern
void sign_padding(unsigned op_bytes, FxpBinaryFuncOpTy& op);

extern
void binary_mul(unsigned lhs_size, FxpBinaryFuncOpTy& lhs_op, int lhs_exp,
                unsigned rhs_size, FxpBinaryFuncOpTy& rhs_op, int rhs_exp,
                unsigned ret_size, FxpBinaryFuncOpTy& ret_val, int& ret_exp);
} // namespace fxp
} // namespace _detail
} // namespace spirv

using namespace spirv;
TEST(FixedPointArithmeticImplTest, BinaryAddBasic){
  /*
   * fixed_point<ful, -2> + fixed_point<ful, -2>
   * -> fixed_point<ful, -2>
   */
  _detail::fxp::FxpBinaryFuncOpTy lhs_op = {9};
  _detail::fxp::FxpBinaryFuncOpTy rhs_op = {4};

  // Output variables
  _detail::fxp::FxpBinaryFuncOpTy ret_val = {0};
  int ret_exponent = 0;

  _detail::fxp::binary_add(4, lhs_op, -2,
                           4, rhs_op, -2,
                           4, ret_val, ret_exponent);
  ASSERT_EQ(ret_exponent, -2);
  ASSERT_EQ(ret_val.Full, 13);
}

TEST(FixedPointArithmeticImplTest, BinaryAddDiffExp){
  /*
   * fixed_point<ful, -3> + fixed_point<ful, -4>
   * -> fixed_point<ful, -4>
   */
  // 8 * 2^-3 = 1
  _detail::fxp::FxpBinaryFuncOpTy lhs_op = {8};
  // 7 * 2^-4 = 7/16
  _detail::fxp::FxpBinaryFuncOpTy rhs_op = {7};

  // Output variables
  _detail::fxp::FxpBinaryFuncOpTy ret_val = {0};
  int ret_exponent = 0;

  _detail::fxp::binary_add(4, lhs_op, -3,
                           4, rhs_op, -4,
                           4, ret_val, ret_exponent);
  // 1 + 7/16 = 23/16 = 23 * 2^-4
  ASSERT_EQ(ret_exponent, -4);
  ASSERT_EQ(ret_val.Full, 23);
}

TEST(FixedPointArithmeticImplTest, BinaryAddDiffWidth){
  /*
   * fixed_point<hlf, -5> + fixed_point<sgl, -5>
   * -> fixed_point<hlf, -5>
   */
  // 9 * 2^-5
  _detail::fxp::FxpBinaryFuncOpTy lhs_op = {9};
  // 4 * 2^-5
  _detail::fxp::FxpBinaryFuncOpTy rhs_op = {4};

  // Output variables
  _detail::fxp::FxpBinaryFuncOpTy ret_val = {0};
  int ret_exponent = 0;

  _detail::fxp::binary_add(2, lhs_op, -5,
                           1, rhs_op, -5,
                           2, ret_val, ret_exponent);
  // 1 + 7/16 = 23/16 = 23 * 2^-4
  ASSERT_EQ(ret_exponent, -5);
  ASSERT_EQ(ret_val.Half, 13);
}

TEST(FixedPointArithmeticImplTest, SignPadding){
  using FxpOpTy = _detail::fxp::FxpBinaryFuncOpTy;
  {
    // Basic
    FxpOpTy u32 = {7};
    _detail::fxp::sign_padding(4, u32);
    ASSERT_EQ(u32.Full, 7);
  }

  {
    // Nearly all 32 bits are filled with 1
    FxpOpTy u8 = {(1 << 30) - 1};
    u8.Single = 7; // 00000111
    _detail::fxp::sign_padding(1, u8);
    ASSERT_EQ(u8.Half, 7);
  }
  
  {
    // All 32 bits are filled with 0
    FxpOpTy s8 = {0};
    s8.Single = 135; // 10000111, -121
    _detail::fxp::sign_padding(1, s8);
    ASSERT_EQ(static_cast<int16_t>(s8.Half), -121);
  }

}

TEST(FixedPointArithmeticImplTest, BinaryMulBasic){
  /*
   * fixed_point<ful, -2> * fixed_point<ful, -2>
   * -> fixed_point<ful, -4>
   */
  // 7 * 2^-2
  _detail::fxp::FxpBinaryFuncOpTy lhs_op = {7};
  // 8 * 2^-2
  _detail::fxp::FxpBinaryFuncOpTy rhs_op = {8};

  // Output variables
  _detail::fxp::FxpBinaryFuncOpTy ret_val = {0};
  int ret_exponent = 0;

  _detail::fxp::binary_mul(4, lhs_op, -2,
                           4, rhs_op, -2,
                           4, ret_val, ret_exponent);
  ASSERT_EQ(ret_exponent, -4);
  ASSERT_EQ(ret_val.Full, 56);
}

TEST(FixedPointArithmeticImplTest, BinaryMulDiffWidthExp){
  /*
   * fixed_point<ful, -2> * fixed_point<hlf, -3>
   * -> fixed_point<ful, -5>
   */
  // 1048575 * 2^-2
  _detail::fxp::FxpBinaryFuncOpTy lhs_op = {(1 << 20) - 1};
  // 32767 * 2^-3
  _detail::fxp::FxpBinaryFuncOpTy rhs_op = {(1 << 15) - 1};

  // Output variables
  _detail::fxp::FxpBinaryFuncOpTy ret_val = {0};
  int ret_exponent = 0;

  _detail::fxp::binary_mul(4, lhs_op, -2,
                           4, rhs_op, -3,
                           4, ret_val, ret_exponent);
  ASSERT_EQ(ret_exponent, -5);
  // Would overflow, silently wrapped
  ASSERT_EQ(ret_val.Full, 4293885953);
}
