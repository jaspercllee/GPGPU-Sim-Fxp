#include <gtest/gtest.h>
#include "spirv_fixed_point.h"

using namespace spirv;

#define THIS_TEST(T) \
  TEST(FixedPointMathFunctionTest, T)

THIS_TEST(IsFixedPointMathFunction) {
  ASSERT_FALSE(IsFxpMathFunction("abcde"));
  ASSERT_FALSE(IsFxpMathFunction(""));

  {
    // Normal case
    typename utils::CXXABIParser::result_type parsed_result;
    ASSERT_TRUE(IsFxpMathFunction("_Z3exp10fxp_8_5_1_", &parsed_result));

    ASSERT_EQ(parsed_result.size(), 2);
    EXPECT_EQ(parsed_result.at(0), "exp");
    EXPECT_EQ(parsed_result.at(1), "fxp_8_5_1_");
  }

  {
    // Function name tailing with number
    typename utils::CXXABIParser::result_type parsed_result;
    ASSERT_TRUE(IsFxpMathFunction("_Z5exp1010fxp_8_5_1_", &parsed_result));

    ASSERT_EQ(parsed_result.size(), 2);
    EXPECT_EQ(parsed_result.at(0), "exp10");
  }
}

THIS_TEST(FxpMathFunctionCreate) {
  auto func = FxpMathFunction::Create("_Z5log1010fxp_8_5_3_");

  ASSERT_TRUE(func->Valid);
  ASSERT_EQ(func->Name, "log10");
  ASSERT_EQ(func->ArgTypes.size(), 1);

  auto* arg_type = dyn_cast<FxpType>(func->ArgTypes.at(0));
  ASSERT_NE(arg_type, nullptr);
  ASSERT_EQ(arg_type->Width, 8);
  ASSERT_EQ(arg_type->Exponent, -5);
  ASSERT_EQ(arg_type->Metadata, 3);
  ASSERT_TRUE(arg_type->Signed);
}
