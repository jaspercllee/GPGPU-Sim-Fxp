#include <gtest/gtest.h>
#include "spirv_fixed_point.h"

using namespace spirv;

#define THIS_TEST(T) \
  TEST(FixedPointConversionTest, T)

THIS_TEST(IsFixedPointConvertFunction) {
  ASSERT_FALSE(IsFxpConvertFunction("abcde"));
  ASSERT_FALSE(IsFxpConvertFunction(""));

  // Integer to fixed-point
  ASSERT_TRUE(IsFxpConvertFunction("_Z24convert_fxp_32_33_3__rtei"));

  // Float to fixed-point
  ASSERT_TRUE(IsFxpConvertFunction("_Z24convert_fxp_16_8_2__rtef"));
  
  // Fixed-point to fixed-point
  ASSERT_TRUE(IsFxpConvertFunction("_Z24convert_fxp_8_7_2__rtnfxp_32_33_3_"));
  
  // Fixed-point to float
  ASSERT_TRUE(IsFxpConvertFunction("_Z17convert_float_rtefxp_32_13_3_"));
}

THIS_TEST(FxpConversionIntToFxp) {
  FxpConvertFunction fxpConv("_Z24convert_fxp_32_33_3__rtei");

  ASSERT_TRUE(fxpConv.Valid);

  // Source type
  ASSERT_TRUE(fxpConv.SrcTy);
  ASSERT_TRUE(dyn_cast<IntType>(fxpConv.SrcTy.get()));

  // Destination type
  ASSERT_TRUE(fxpConv.DestTy);
  ASSERT_TRUE(dyn_cast<FxpType>(fxpConv.DestTy.get()));
}

THIS_TEST(FxpConversionFloatToFxp) {
  FxpConvertFunction fxpConv("_Z24convert_fxp_16_8_2__rtef");

  ASSERT_TRUE(fxpConv.Valid);

  // Source type
  ASSERT_TRUE(fxpConv.SrcTy);
  ASSERT_TRUE(dyn_cast<FloatType>(fxpConv.SrcTy.get()));

  // Destination type
  ASSERT_TRUE(fxpConv.DestTy);
  ASSERT_TRUE(dyn_cast<FxpType>(fxpConv.DestTy.get()));
}

THIS_TEST(FxpConversionFxpToFloat) {
  FxpConvertFunction fxpConv("_Z17convert_float_rtefxp_32_13_3_");

  ASSERT_TRUE(fxpConv.Valid);

  // Source type
  ASSERT_TRUE(fxpConv.SrcTy);
  ASSERT_TRUE(dyn_cast<FxpType>(fxpConv.SrcTy.get()));

  // Destination type
  ASSERT_TRUE(fxpConv.DestTy);
  ASSERT_TRUE(dyn_cast<FloatType>(fxpConv.DestTy.get()));
}
