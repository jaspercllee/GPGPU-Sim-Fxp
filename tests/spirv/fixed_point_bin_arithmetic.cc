#include <gtest/gtest.h>
#include "spirv_fixed_point.h"

using namespace spirv;

TEST(FixedPointBinArithmeticTest, IsFixedPointBinaryFunction) {
  
  // Obviously error
  ASSERT_FALSE(IsFxpBinaryFunction("abcdefg"));

  // Arithmetic Division
  ASSERT_TRUE(IsFxpBinaryFunction("_Z7fxp_div12fxp_32_28_3_12fxp_32_28_3_"));

  // Arithmetic Multiplication
  ASSERT_TRUE(IsFxpBinaryFunction("_Z7fxp_mul12fxp_32_28_3_12fxp_32_28_3_"));

  // Arithmetic Addition(legacy)
  ASSERT_FALSE(IsFxpBinaryFunction("_Z14__spirv_FxpAddPU3AS425__spirv_FixedPoint__8_3_3PU3AS426__spirv_FixedPoint__16_3_3"));
}

TEST(FixedPointBinArithmeticTest, FxpBinaryFunctionWithAddOp){
  FxpBinaryFunction fxpFunc("_Z7fxp_add12fxp_8_3_3_12fxp_16_3_3_");

  ASSERT_TRUE(fxpFunc.Valid);

  ASSERT_EQ(fxpFunc.OpName, "add");

  ASSERT_EQ(fxpFunc.LhsTotalBits, 8);
  ASSERT_EQ(fxpFunc.LhsExponent, -3);
  ASSERT_EQ(fxpFunc.RhsTotalBits, 16);
  ASSERT_EQ(fxpFunc.RhsExponent, -3);
}
TEST(FixedPointBinArithmeticTest, FxpBinaryFunctionWithSubOp){
  FxpBinaryFunction fxpFunc("_Z7fxp_sub12fxp_8_3_3_12fxp_16_4_3_");

  ASSERT_TRUE(fxpFunc.Valid);

  ASSERT_EQ(fxpFunc.OpName, "sub");

  ASSERT_EQ(fxpFunc.LhsTotalBits, 8);
  ASSERT_EQ(fxpFunc.LhsExponent, -3);
  ASSERT_EQ(fxpFunc.RhsTotalBits, 16);
  ASSERT_EQ(fxpFunc.RhsExponent, -4);
}
TEST(FixedPointBinArithmeticTest, FxpBinaryFunctionWithMulOp){
  FxpBinaryFunction fxpFunc("_Z7fxp_mul12fxp_8_3_3_12fxp_8_5_1_");

  ASSERT_TRUE(fxpFunc.Valid);

  ASSERT_EQ(fxpFunc.OpName, "mul");

  ASSERT_EQ(fxpFunc.LhsTotalBits, 8);
  ASSERT_EQ(fxpFunc.LhsExponent, -3);
  ASSERT_EQ(fxpFunc.RhsTotalBits, 8);
  ASSERT_EQ(fxpFunc.RhsExponent, 5);
}
TEST(FixedPointBinArithmeticTest, FxpBinaryFunctionWithDivOp){
  FxpBinaryFunction fxpFunc("_Z7fxp_div12fxp_32_33_2_12fxp_8_5_1_");

  ASSERT_TRUE(fxpFunc.Valid);

  ASSERT_EQ(fxpFunc.OpName, "div");

  ASSERT_EQ(fxpFunc.LhsTotalBits, 32);
  ASSERT_EQ(fxpFunc.LhsExponent, -33);
  ASSERT_EQ(fxpFunc.RhsTotalBits, 8);
  ASSERT_EQ(fxpFunc.RhsExponent, 5);
}
