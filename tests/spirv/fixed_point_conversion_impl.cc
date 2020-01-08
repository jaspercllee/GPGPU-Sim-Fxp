#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include <cstdint>
#include "spirv_fixed_point.h"

namespace spirv {
namespace _detail {

template<unsigned SrcWidth, unsigned DestWidth, 
         typename SrcTy = typename FxpWidth<SrcWidth>::type,
         typename DestTy = typename FxpWidth<DestWidth>::type>
void PerformFxp2Fxp(const SrcTy& src_val, DestTy& dest_val, 
                    int SrcExponent, int DestExponent,
                    bool is_sign,
                    RoundingMode Rounding, bool Saturation);

void extractFloat(float val, 
                  uint32_t& significand, int32_t& exponent, 
                  bool& is_sign);

template<unsigned Width,
         typename SignifTy = typename FxpWidth<Width>::type>
void fxpToFloat(SignifTy significand, int exponent, bool isSigned,
                float& dest_value);

} // end namespace _detail
} // end namespace spirv

using namespace spirv;

#define THIS_TEST(T) \
  TEST(FixedPointConversionImplTest, T)

inline
bool fequal(float a, float b){
  float delta = ::fabs(a - b);
  return delta < std::numeric_limits<float>::epsilon();
}

THIS_TEST(ExtractFloat) {
  uint32_t significand;
  int32_t exponent;
  bool is_sign;
  auto isEqual = [&](float origin_val, float& val) {
    val = ((float)significand) * ::pow(2.0f, (float)exponent);
    val = is_sign? -val : val;
    return fequal(origin_val, val);
  };

  float origin_val;
  float val;
  
  origin_val = -9.87f;
  _detail::extractFloat(origin_val, 
                        significand, exponent, is_sign);
  ASSERT_TRUE(is_sign);
  EXPECT_TRUE(isEqual(origin_val, val)) 
              << "Val: " << val;

  origin_val = 3.14f;
  _detail::extractFloat(origin_val, 
                        significand, exponent, is_sign);
  ASSERT_FALSE(is_sign);
  EXPECT_TRUE(isEqual(origin_val, val)) 
              << "Val: " << val;
  
  origin_val = 24.445f;
  _detail::extractFloat(origin_val, 
                        significand, exponent, is_sign);
  ASSERT_FALSE(is_sign);
  EXPECT_TRUE(isEqual(origin_val, val)) 
              << "Val: " << val;
  
  origin_val = -94.87f;
  _detail::extractFloat(origin_val, 
                        significand, exponent, is_sign);
  ASSERT_TRUE(is_sign);
  EXPECT_TRUE(isEqual(origin_val, val)) 
              << "Val: " << val;
  
  origin_val = 94.87f;
  _detail::extractFloat(origin_val, 
                        significand, exponent, is_sign);
  ASSERT_FALSE(is_sign);
  EXPECT_TRUE(isEqual(origin_val, val)) 
              << "Val: " << val;
}

THIS_TEST(Fxp2FxpTrivial) {
  uint32_t src_val = 0x10;
  uint16_t dest_val;
  int src_exponent = 5;
  int dest_exponent = 5;

  _detail::PerformFxp2Fxp<32,16>(src_val, dest_val,
                                 src_exponent, dest_exponent,
                                 false,
                                 RoundingMode::rte, false);
  ASSERT_TRUE(dest_val == 0x10);
}

THIS_TEST(Fxp2FxpRoundingRTE) {
  uint32_t src_val = (1 << 16) - 1; // 65535
  uint16_t dest_val;
  int src_exponent = 5;
  int dest_exponent = 10;

  _detail::PerformFxp2Fxp<32,16>(src_val, dest_val,
                                 src_exponent, dest_exponent,
                                 false,
                                 RoundingMode::rte, false);
  ASSERT_EQ(dest_val, 2048);
}

THIS_TEST(Fxp2Float) {
  uint32_t src_val = (1 << 16) - 1; // 65535
  constexpr int exponent = -2;

  float dest_val;
  _detail::fxpToFloat<32>(src_val, exponent, false,
                          dest_val);
  ASSERT_TRUE(fequal(dest_val, 16383.75f))
            << "Val: " << dest_val;
}
