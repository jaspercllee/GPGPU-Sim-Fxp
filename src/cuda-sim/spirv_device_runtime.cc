#include <cstdint>
#include <cctype>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <functional>

#include "../gpgpu-sim/gpu-sim.h"
#include "cuda-sim.h"
#include "ptx_ir.h"
#include "spirv_device_runtime.h"

#define DEV_RUNTIME_REPORT(a) \
   if( g_debug_execution ) { \
      std::cout << __FILE__ << ", " << __LINE__ << ": " << a << "\n"; \
      std::cout.flush(); \
   }

namespace spirv {
namespace _detail {

/*
 * Utility routines
 */
static inline unsigned s2u_abs(int S){
  return (S >= 0)? (unsigned)S : (unsigned)-S;
}
// Immutable transform
static inline std::string str_tolower(const std::string& src){
  std::stringstream ss;
  unsigned i;
  for(i = 0; i < src.length(); ++i)
    ss << (char)std::tolower(src[i]);
  
  return ss.str();
}

namespace fxp{
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

using thread_ctx_info = 
  std::tuple<ptx_thread_info*, const function_info*, const ptx_instruction*>;

void ExtractArgInfo(const ptx_instruction* pI, const function_info* target_func,
                    unsigned index, 
                    unsigned& size, addr_t& from_addr) { 
    unsigned n_return = target_func->has_return();
    assert(n_return);
    const operand_info &actual_param_op = pI->operand_lookup(n_return + 1 + index); //param#
    const symbol *formal_param = target_func->get_arg(index);

    size = formal_param->get_size_in_bytes();
    assert( formal_param->is_param_local() );
    assert( actual_param_op.is_param_local() );

    from_addr = actual_param_op.get_symbol()->get_address();
}

template<typename T>
inline
void GeneralFetchOperand(ptx_thread_info* thread, 
                         unsigned byte_size, const addr_t& from_addr, 
                         T& output){
  thread->m_local_mem->read(from_addr, byte_size, &output);
}
template<typename T>
inline
void GeneralWriteReturn(const thread_ctx_info& ctx, 
                        const T& result_val) {
  const ptx_instruction *pI;
  ptx_thread_info* thread;
  const function_info *target_func;
  std::tie(thread, target_func, pI) = ctx;

  const operand_info &actual_return_op = pI->operand_lookup(0); //retval0
  addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
  const symbol *formal_return = target_func->get_return_var();
  unsigned int return_size = formal_return->get_size_in_bytes(); 
  thread->m_local_mem->write(ret_param_addr, return_size, &result_val,
                             nullptr, nullptr);
}

inline
void roundingRTE(unsigned DestWidth, unsigned scale,
                 uint32_t& general_dest_val){
  if(scale >= DestWidth){
    // Assign zero directly
    general_dest_val = 0;
  }else{
    uint32_t bias = 1 << (scale - 1);
    general_dest_val += bias;
    general_dest_val >>= scale;
  }
}

inline
void unconditionalAdvance(unsigned DestWidth, unsigned scale,
                          uint32_t& general_dest_val){
  if(scale >= DestWidth){
    // Assign zero directly
    general_dest_val = 0;
  }else{
    uint32_t bias = 1 << scale;
    general_dest_val += bias;
    general_dest_val >>= scale;
  }
}
inline
void truncate(unsigned scale, 
              uint32_t& general_dest_val){
  general_dest_val >>= scale;
}

inline
void roundingRTZ(unsigned scale, 
                 uint32_t& general_dest_val){
  truncate(scale, general_dest_val);
}

inline
void roundingRTP(unsigned DestWidth, unsigned scale,
                 uint32_t& general_dest_val,
                 bool isSigned){
  // Negative
  if(isSigned) truncate(scale, general_dest_val);
  // Positive
  else unconditionalAdvance(DestWidth, scale, general_dest_val);
}

inline
void roundingRTN(unsigned DestWidth, unsigned scale,
                 uint32_t& general_dest_val, 
                 bool isSigned){
  // Negative
  if(isSigned) unconditionalAdvance(DestWidth, scale, general_dest_val);
  // Positive
  else truncate(scale, general_dest_val);
}

template<unsigned SrcWidth, unsigned DestWidth, 
         typename SrcTy = typename FxpWidth<SrcWidth>::type,
         typename DestTy = typename FxpWidth<DestWidth>::type>
void PerformFxp2Fxp(const SrcTy& src_val, DestTy& dest_val, 
                    int SrcExponent, int DestExponent,
                    bool is_sign,
                    RoundingMode Rounding, bool Saturation){
  int exponent_delta = DestExponent - SrcExponent;
  unsigned scale = (unsigned)::abs(exponent_delta);

  // Both transform to max width type(i.e 32bits) in our system
  assert(SrcWidth <= 32 && DestWidth <= 32);
  uint32_t general_src_val = (uint32_t)src_val;
  uint32_t general_dest_val = general_src_val;
  uint32_t max_dest_val = (uint32_t)FxpWidth<DestWidth>::max;
  if(exponent_delta < 0){
    // Left shift to enlarge the significand part
    // May overflow 
    unsigned i = 0;
    for(i = 0; i < scale; ++i){
      if(general_dest_val >= max_dest_val && Saturation){
        general_dest_val = max_dest_val;
        break;
      }
      general_dest_val <<= 1;
    }
  }else if(exponent_delta > 0){
    // Right shift to reduce the significand part
    // May need rounding
    switch(Rounding){
      case RoundingMode::rte:
        roundingRTE(DestWidth, scale, 
                    general_dest_val);
        break;
      case RoundingMode::rtz:
        roundingRTZ(scale,
                    general_dest_val);
        break;
      case RoundingMode::rtp:
        roundingRTP(DestWidth, scale, 
                    general_dest_val,
                    is_sign);
        break;
      case RoundingMode::rtn:
        roundingRTN(DestWidth, scale, 
                    general_dest_val,
                    is_sign);
        break;
    }
    if(exponent_delta >= 32)  general_dest_val = 0; // fix shift more than 32 bits
    if(general_dest_val >= max_dest_val && Saturation)
      general_dest_val = max_dest_val;
  }else { // exponent_delta == 0
    // Watch out src_val might not fit in destination
    if(general_dest_val >= max_dest_val && Saturation)
      general_dest_val = max_dest_val;
  }
  
  // Mask out the desired value
  general_dest_val = general_dest_val & max_dest_val;
  dest_val = (DestTy)general_dest_val;
}

constexpr uint32_t FLOAT_EXP_BIT_MASK = 0x7F800000;
constexpr uint32_t FLOAT_SIGNIFICAND_WIDTH = 24;
constexpr uint32_t FLOAT_EXP_BIAS = 127;
constexpr uint32_t FLOAT_SIGNIF_BIT_MASK = 0x007FFFFF;
constexpr uint32_t FLOAT_SIGN_BIT_MASK = 0x80000000;

// Workaround: Outline this function for the sake of unit testing
// **DO NOT MARK INLINE**
void extractFloat(float val, 
                  uint32_t& significand, int32_t& exponent, 
                  bool& is_sign){
  // Get exponent and significand value
  if( 0.00000001 > val && val > -0.00000001 )
  {
    if(val > 0)
    {
      //val = FLT_MIN;
      //val = 0.00001;
      //val = 0.000001;
      //val = 0.0000001;
      //val = 0.00000001;
      //val = 0.000000001;
      //val = 0.0000000001;
      //val = 0.00000000001;
      //val = 0.000000000001;
      //val = 0.00000000000001; xx

    }
    else if(val < 0)
    {
      val = FLT_MIN*(-1);
    } 
    //raw_bits = 0;
  }
  uint32_t raw_bits;
  static_assert(sizeof(val) == sizeof(raw_bits), 
                "width of float is not 32-bits?!");
  ::memcpy(&raw_bits, &val, sizeof(raw_bits));
  uint32_t tmp = static_cast<uint32_t>(val);
  if( 0.00000001 > val && val > -0.00000001 )
  {
    //raw_bits = 0;
  }
  exponent = (int32_t)( (int32_t)((raw_bits & FLOAT_EXP_BIT_MASK) >>  
                         (FLOAT_SIGNIFICAND_WIDTH - 1)) - (int32_t)FLOAT_EXP_BIAS );
  significand = (uint32_t)(raw_bits & FLOAT_SIGNIF_BIT_MASK);
  // Append the implicit bit
  significand |= (1 << (FLOAT_SIGNIFICAND_WIDTH - 1));
  // Rescale the exponent
  exponent -= (int)(FLOAT_SIGNIFICAND_WIDTH - 1);
  is_sign = (bool)(raw_bits & FLOAT_SIGN_BIT_MASK);
}

template<typename T>
T twosComplement(T val){
  return (~val) + static_cast<T>(0x01);
}

template<unsigned Width,
         typename SignifTy = typename FxpWidth<Width>::type>
void fxpToFloat(SignifTy significand, int exponent, bool isSigned,
                float& dest_value){
  if(isSigned){
    using SSignifTy = typename std::make_signed<SignifTy>::type;
    SSignifTy val = static_cast<SSignifTy>(significand);
    dest_value = ((float)val) * ::pow(2.0f, (float)exponent);
  }else
    dest_value = ((float)significand) * ::pow(2.0f, (float)exponent);
}

template <unsigned IntW,
          typename IntTy = typename IntWidth<IntW>::type,
          typename IntValTy = typename IntWidth<IntW>::value_type,
          typename UIntValTy = typename std::make_unsigned<IntValTy>::type>
void ConvertInteger2Fxp(const thread_ctx_info& ctx, 
                        const IntTy* src_type, const FxpType* dest_type,
                        RoundingMode RoundMode, bool Saturation){
  const ptx_instruction *pI;
  ptx_thread_info* thread;
  const function_info *target_func;
  std::tie(thread, target_func, pI) = ctx;
  
  addr_t src_addr;
  unsigned src_size;
  _detail::ExtractArgInfo(pI, target_func, 0, src_size, src_addr);
  //assert(src_type->Size == (src_size << 3));
  assert(src_type->Size <= IntW);
  
  IntValTy src_value;
  GeneralFetchOperand(thread, src_size, src_addr, src_value);
  
  bool is_neg = src_value < 0;
  assert((dest_type->Signed || !is_neg) &&
         "Can not convert signed to unsigned");
  UIntValTy abs_value = is_neg? (UIntValTy)-src_value : (UIntValTy)src_value;

  switch(dest_type->Width){
    case 8:{
      uint8_t dest_val;
      PerformFxp2Fxp<IntW,8>(abs_value, dest_val, 
                             0, dest_type->Exponent,
                             is_neg,
                             RoundMode, Saturation);
      if(is_neg) dest_val = twosComplement(dest_val);
      GeneralWriteReturn(ctx, dest_val);
      break;
    }
    case 16:{
      uint16_t dest_val;
      PerformFxp2Fxp<IntW,16>(abs_value, dest_val, 
                              0, dest_type->Exponent,
                              is_neg,
                              RoundMode, Saturation);
      if(is_neg) dest_val = twosComplement(dest_val);
      GeneralWriteReturn(ctx, dest_val);
      break;
    }
    case 32:{
      uint32_t dest_val;
      PerformFxp2Fxp<IntW,32>(abs_value, dest_val, 
                              0, dest_type->Exponent,
                              is_neg,
                              RoundMode, Saturation);
      if(is_neg) dest_val = twosComplement(dest_val);
      GeneralWriteReturn(ctx, dest_val);
      break;
    }
    default:
      assert(false && "Unsupported width");
  }
}

void ConvertFxp2Float(const thread_ctx_info& ctx, 
                      const FxpType* src_type, const FloatType* dest_type) {
  const ptx_instruction *pI;
  ptx_thread_info* thread;
  const function_info *target_func;
  std::tie(thread, target_func, pI) = ctx;
  
  addr_t src_addr;
  unsigned src_size;
  _detail::ExtractArgInfo(pI, target_func, 0, src_size, src_addr);
  //assert(src_type->Size == (src_size << 3));
  assert(src_type->Size <= 32);

  float dest_value;
#define Case(W) \
    case (W):{  \
      using USignTy = typename UIntWidth<W>::value_type;  \
      USignTy src_value;  \
      GeneralFetchOperand(thread, src_size, src_addr, src_value); \
      fxpToFloat<W>(src_value, src_type->Exponent, src_type->Signed,  \
                    dest_value);  \
      break;  \
    }

  switch(src_type->Width){
    Case(8)
    Case(16)
    Case(32)
    default:
      assert(false && "Unsupported width");
  }
  GeneralWriteReturn(ctx, dest_value);
#undef Case
}

void ConvertFloat2Fxp(const thread_ctx_info& ctx,
                      const FloatType* src_type, const FxpType* dest_type,
                      RoundingMode RoundMode, bool Saturation){
  const ptx_instruction *pI;
  ptx_thread_info* thread;
  const function_info *target_func;
  std::tie(thread, target_func, pI) = ctx;
  
  addr_t src_addr;
  unsigned src_size;
  _detail::ExtractArgInfo(pI, target_func, 0, src_size, src_addr);
  assert(src_type->Size == (src_size << 3));
  assert(src_type->Size <= 32);
  float src_value;
  GeneralFetchOperand(thread, src_size, src_addr, src_value);
  
  float tmp = 0.0;
  if( src_value )
  {
    if(src_value == tmp)
    {
      int a = 0;
      src_value = (float)a;
    }

  }

  uint32_t significand;
  int32_t exponent;
  bool is_negative;
  extractFloat(src_value, significand, exponent, is_negative);
  // FIXME: There is a serious mis-understanding between compiler / lib
  // side and simulator. Where the former denote "IsSigned" as
  // "using signed integer as mantissa", and the latter one denote as
  // "mantissa is negative"
  //assert(is_sign == dest_type->Signed && 
         //"Signess mismatch");
  assert((dest_type->Signed || !is_negative) &&
         "Can not convert signed to unsigned");

#define Case(W) \
    case (W):{  \
      typename UIntWidth<W>::value_type dest_val; \
      PerformFxp2Fxp<32,W>(significand, dest_val, \
                           (int)exponent, dest_type->Exponent,  \
                           is_negative, \
                           RoundMode, Saturation);  \
      if(is_negative) dest_val = twosComplement(dest_val);  \
      if(src_value == 0)  dest_val = 0; \
      GeneralWriteReturn(ctx, dest_val);  \
      break;  \
    }

  switch(dest_type->Width){
    Case(8)
    Case(16)
    Case(32)
    default:
      assert(false && "Unsupported width");
  }
#undef Case
}

// Materialize src / dest width
// Mask layout:
//    Dest    Src
// |32|16|8|32|16|8|
#define MT(SrcW, DestW) ((DestW) | (SrcW) >> 3) 
void ConvertFxp2Fxp(const thread_ctx_info& ctx, 
                    const FxpType* src_type, const FxpType* dest_type,
                    RoundingMode Rounding, bool Saturation) {
  const ptx_instruction *pI;
  ptx_thread_info* thread;
  const function_info *target_func;
  std::tie(thread, target_func, pI) = ctx;
  
  addr_t src_addr;
  unsigned src_size;
  _detail::ExtractArgInfo(pI, target_func, 0, src_size, src_addr);
  assert(src_type->Size <= 32 && dest_type->Size <= 32);

  assert(!(src_type->Signed && !dest_type->Signed) &&
         "Can not convert signed to unsigned");

#define Case(DestW, SrcW) \
case MT(SrcW, DestW): { \
  using SrcTy = typename FxpWidth<SrcW>::type; \
  using SSrcTy = std::make_signed<SrcTy>::type; \
  using DestTy = typename FxpWidth<DestW>::type; \
  SrcTy src_value;  \
  DestTy dest_value;  \
  GeneralFetchOperand(thread, src_size, src_addr, src_value); \
  bool is_neg = false;  \
  if(src_type->Signed){ \
    SSrcTy sign_src_value = static_cast<SSrcTy>(src_value); \
    is_neg = sign_src_value < 0;  \
    if(is_neg) src_value = static_cast<SrcTy>(-sign_src_value); \
  } \
  PerformFxp2Fxp<SrcW, DestW>(src_value, dest_value, \
                              src_type->Exponent, dest_type->Exponent, \
                              is_neg, \
                              Rounding, Saturation); \
  if(is_neg) dest_value = twosComplement(dest_value); \
  GeneralWriteReturn(ctx, dest_value); \
  break; \
}

  switch( MT(src_type->Width, dest_type->Width) ){
    Case(8,8)
    Case(16,16)
    Case(32,32)
    Case(8,16)
    Case(8,32)
    Case(16,32)
    Case(16,8)
    Case(32,8)
    Case(32,16)
    default:
      assert(false && "Unsupported width");
  }
#undef Case
}
#undef MT

// Math function with only one argument
// FIXME: What about rounding mode and saturation?
template<typename FuncTy>
void DoSimpleFxpMath(const thread_ctx_info& ctx, 
                     const FxpType* arg_type,
                     FuncTy math_func,
                     RoundingMode RoundMode = RoundingMode::rtn,
                     bool Saturation = false) {
  const ptx_instruction *pI;
  ptx_thread_info* thread;
  const function_info *target_func;
  std::tie(thread, target_func, pI) = ctx;
  
  addr_t arg_addr;
  unsigned arg_size;
  _detail::ExtractArgInfo(pI, target_func, 0, arg_size, arg_addr);
  //assert(arg_type->Size == (arg_size << 3));
  assert(arg_type->Size <= 32);

  float tmp_value, tmp_result;
  uint32_t tmp_signif;
  int32_t tmp_exp;
  bool tmp_sign;
#define CASE(T,W) \
  case (W):{  \
    T arg_value, dest_value;  \
    GeneralFetchOperand(thread, arg_size, arg_addr, arg_value); \
    fxpToFloat<(W)>(arg_value, arg_type->Exponent, arg_type->Signed,  \
                    tmp_value); \
    \
    tmp_result = math_func(tmp_value);  \
    \
    extractFloat(tmp_result, tmp_signif, tmp_exp, tmp_sign);  \
    PerformFxp2Fxp<32,(W)>(tmp_signif, dest_value, \
                         (int)tmp_exp, arg_type->Exponent,  \
                         tmp_sign,  \
                         RoundMode, Saturation);  \
    GeneralWriteReturn(ctx, dest_value);  \
    break;  \
  }
  switch(arg_type->Width){
    CASE(uint8_t, 8)
    CASE(uint16_t, 16)
    CASE(uint32_t, 32)
    default:
      assert(false && "Unsupported width");
  }
#undef CASE
}
} // namespace _detail
} // namespace spirv

using namespace spirv;
void gpgpusim_spirv_invokeFxpConvertFunction(const ptx_instruction* pI,
                                            ptx_thread_info* thread, const function_info* target_func,
                                            const FxpConvertFunction& fxp_convert_func){
    assert(fxp_convert_func.Valid &&
           "Not a valid fxp conversion function");
    
    //unsigned n_return = target_func->has_return();
    unsigned n_args = target_func->num_args();
    assert( n_args == 1 );

    auto& src_type = fxp_convert_func.SrcTy;
    auto& dest_type = fxp_convert_func.DestTy;
    assert((src_type->Tag == TypeTag::FixedPoint || 
            dest_type->Tag == TypeTag::FixedPoint) &&
           "Either source or destination type need to be fixed-point");

    auto thread_ctx = std::make_tuple(thread, target_func, pI);
    if(src_type->Tag == TypeTag::Float &&
       dest_type->Tag == TypeTag::FixedPoint){
      _detail::ConvertFloat2Fxp(thread_ctx, 
                                dyn_cast<FloatType>(src_type.get()), 
                                dyn_cast<FxpType>(dest_type.get()), 
                                fxp_convert_func.RoundMode,
                                fxp_convert_func.Saturation);
    } else if(src_type->Tag == TypeTag::Integer &&
              dest_type->Tag == TypeTag::FixedPoint){
      _detail::ConvertInteger2Fxp<32>(thread_ctx, 
                                      dyn_cast<IntType>(src_type.get()), 
                                      dyn_cast<FxpType>(dest_type.get()), 
                                      fxp_convert_func.RoundMode,
                                      fxp_convert_func.Saturation);
    } else if(src_type->Tag == TypeTag::Integer16 &&
              dest_type->Tag == TypeTag::FixedPoint){
      _detail::ConvertInteger2Fxp<16>(thread_ctx, 
                                      dyn_cast<Int16Type>(src_type.get()), 
                                      dyn_cast<FxpType>(dest_type.get()), 
                                      fxp_convert_func.RoundMode,
                                      fxp_convert_func.Saturation);
    } else if(src_type->Tag == TypeTag::Integer8 &&
              dest_type->Tag == TypeTag::FixedPoint){
      _detail::ConvertInteger2Fxp<8>(thread_ctx, 
                                     dyn_cast<Int8Type>(src_type.get()), 
                                     dyn_cast<FxpType>(dest_type.get()), 
                                     fxp_convert_func.RoundMode,
                                     fxp_convert_func.Saturation);
    } else if(src_type->Tag == TypeTag::FixedPoint &&
              dest_type->Tag == TypeTag::Float){
      _detail::ConvertFxp2Float(thread_ctx, 
                                dyn_cast<FxpType>(src_type.get()), 
                                dyn_cast<FloatType>(dest_type.get()));
    }else if(src_type->Tag == TypeTag::FixedPoint &&
             dest_type->Tag == TypeTag::FixedPoint){
      _detail::ConvertFxp2Fxp(thread_ctx, 
                              dyn_cast<FxpType>(src_type.get()), 
                              dyn_cast<FxpType>(dest_type.get()),
                              fxp_convert_func.RoundMode,
                              fxp_convert_func.Saturation);
    }else{
      assert(false && "Unimplemented");
    }
}

void gpgpusim_spirv_invokeFxpBinaryFunction(const ptx_instruction* pI, 
                                            ptx_thread_info* thread, const function_info* target_func,
                                            const FxpBinaryFunction& fxp_binary_func){
    assert(fxp_binary_func.Valid &&
           "Not a valid fxp binary function");

    DEV_RUNTIME_REPORT("Calling SPIRV Fxp binary function: "
                       << fxp_binary_func);

    unsigned n_args = target_func->num_args();
    assert( n_args == 2 );

    auto fetchOperand = [thread](unsigned size, const addr_t& from_addr, 
                                 _detail::fxp::FxpBinaryFuncOpTy& operand) {
      switch(size << 3){
        default:
          assert(false && "Only support 8/16/32 bits for now");
        case 8:
          _detail::GeneralFetchOperand(thread, size, from_addr, operand.Single);
          break;

        case 16:
          _detail::GeneralFetchOperand(thread, size, from_addr, operand.Half);
          break;

        case 32:
        
          _detail::GeneralFetchOperand(thread, size, from_addr, operand.Full);
          break;
      }
    };

    unsigned lhs_size, rhs_size;
    addr_t lhs_addr, rhs_addr;
    _detail::fxp::FxpBinaryFuncOpTy lhs_op, rhs_op;
    // Arg0: LHS operand
    _detail::ExtractArgInfo(pI, target_func, 0, lhs_size, lhs_addr);
    /*
    assert((fxp_binary_func.LhsTotalBits >> 3) == lhs_size &&
           "LHS operand total width mismatch");
           */
    //fetchOperand(lhs_size, lhs_addr, lhs_op);
    fetchOperand((fxp_binary_func.LhsTotalBits >> 3), lhs_addr, lhs_op);

    // Arg1: RHS operand
    _detail::ExtractArgInfo(pI, target_func, 1, rhs_size, rhs_addr);
    /*
    assert((fxp_binary_func.RhsTotalBits >> 3) == rhs_size &&
           "RHS operand total width mismatch");
           */
    //fetchOperand(rhs_size, rhs_addr, rhs_op);
    fetchOperand((fxp_binary_func.RhsTotalBits >> 3), rhs_addr, rhs_op);
    

    const operand_info &actual_return_op = pI->operand_lookup(0); //retval0
    addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
    const symbol *formal_return = target_func->get_return_var();
    unsigned int return_size = formal_return->get_size_in_bytes();
    
    const auto& op_name = _detail::str_tolower(fxp_binary_func.OpName);

#define SELECT_OP(operation) \
    if(op_name == #operation ){ \
      _detail::fxp::FxpBinaryFuncOpTy return_val = {0}; \
      int return_exp = 0; \
      _detail::fxp::binary_##operation(lhs_size, lhs_op, fxp_binary_func.LhsExponent,\
                                       rhs_size, rhs_op, fxp_binary_func.RhsExponent, \
                                       return_size, return_val, return_exp); \
      FXP_EXTRACT(return_size, return_val, { \
        thread->m_local_mem->write(ret_param_addr, return_size, &val, \
                                   nullptr, nullptr); \
      }) \
    }else

    SELECT_OP(add)
    SELECT_OP(mul)
    {
      assert(false && "Operation not supported yet");
    }
    // TODO: Deligate to operation handlers
}

void gpgpusim_spirv_invokeFxpMathFunction(const ptx_instruction* pI,
                                          ptx_thread_info* thread, 
                                          const function_info* target_func,
                                          const FxpMathFunction& fxp_math_func){
  assert(fxp_math_func.Valid &&
         "Invalid math function format");

  auto thread_ctx = std::make_tuple(thread, target_func, pI);
  if(fxp_math_func.Name == "exp"){
    assert(fxp_math_func.ArgTypes.size() >= 1 &&
           "Insufficient argument amount");
    _detail::DoSimpleFxpMath(thread_ctx, 
                             dyn_cast<FxpType>(fxp_math_func.ArgTypes.at(0)),
                             [](float x)->float {
                              return std::exp(x);
                             });
  }else if(fxp_math_func.Name == "log"){
    assert(fxp_math_func.ArgTypes.size() >= 1 &&
           "Insufficient argument amount");
    _detail::DoSimpleFxpMath(thread_ctx, 
                             dyn_cast<FxpType>(fxp_math_func.ArgTypes.at(0)),
                             [](float x)->float {
                              return std::log(x);
                             });
  }else
    assert(false && "Math function not supported yet");
}
