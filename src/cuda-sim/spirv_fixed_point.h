#pragma once
#include <regex>
#include <memory>
#include <algorithm>

#include <boost/spirit/include/qi.hpp>

#include "spirv_common.h"
#include "spirv_abi_parser.h"

namespace spirv {
namespace _detail{
namespace fxp{
using namespace boost::spirit;
using qi::uint_;
using qi::string;
using qi::lit;

/*
 * At first we want to share this rule with ABI string parsing
 * part within FxpBinaryFunction ctor, we hope to use the following
 * variant of qi::parse to fetch placeholders' values:
 * qi::parse(begin, end, BINARY_FUNC_RULES,
 *           unused,
 *           OpName,
 *           unused, unused,
 *           LhsTotalBits, LhsExponent, LhsMetadata, ...)
 * However, it since that the implementation of qi::parse can only 
 * accomodate limited amount of vararg("receiver" variable
 * for each placeholder value)
 */
#define SUB_RULE_FXP() \
    ( "fxp_" \
    >> uint_ >> '_' \
    >> uint_ >> '_' \
    >> uint_ >> '_' )

const auto BINARY_FUNC_RULES = qi::copy(
    "_Z" >> uint_ >>
    "fxp_"
    >> (string("add") | string("sub") | string("mul") | string("div"))
    >> uint_
    >> SUB_RULE_FXP()
    >> uint_
    >> SUB_RULE_FXP()
);

const auto CONVERT_FUNC_RULES = qi::copy(
    "_Z" >> uint_
    >> "convert_"
    >> (SUB_RULE_FXP() |
        string("int") | string("uint") |
        string("short") | string("ushort") |
        string("char") | string("uchar") |
        string("float") | string("double"))
    >> (string("_rte") | string("_rtn") | 
        string("_rtz") | string("_rtp"))
    >> (lit('c') | // char
        lit('h') | // uchar
        lit('t') | // ushort
        lit('s') | // short
        lit('j') | // uint
        lit('i') | // int
        lit('m') | // ulong
        lit('l') | // long
        lit('f') | // float
        lit('d') | // double
        SUB_RULE_FXP() )
);

typedef union {
  // Full must be the first member
  // for the sake of neat initialization
  uint32_t Full;
  uint16_t Half;
  uint8_t Single;
} FxpBinaryFuncOpTy;

} // namespace fxp
} // namespace _detail

  inline bool IsFxpBinaryFunction(const std::string& input){
    return boost::spirit::qi::parse(
      input.begin(), input.end(),
      _detail::fxp::BINARY_FUNC_RULES
    );
  }

  constexpr unsigned FXP_META_MASK_ACU_SIGNED =     (1 << 0);
  constexpr unsigned FXP_META_MASK_EXP_NEGATIVE =   (1 << 1);

  struct FxpBinaryFunction {
    bool Valid;
    std::string OpName;

    unsigned LhsTotalBits,
             LhsMetadata;
    int LhsExponent;

    unsigned RhsTotalBits,
             RhsMetadata;
    int RhsExponent;

    FxpBinaryFunction(const std::string& input);

    friend std::ostream& operator<<(std::ostream& OS, const FxpBinaryFunction& FxpBFunc);

  private:

    // Placeholder for unused parsed results
    int ph_unused_;

    // Intermediate result for exponent
    unsigned lhs_exponent_, rhs_exponent_;

    //std::smatch match_result_;
  };
 
  // Convert functions
  inline bool IsFxpConvertFunction(const std::string& input){
    return boost::spirit::qi::parse(
      input.begin(), input.end(),
      _detail::fxp::CONVERT_FUNC_RULES
    );
  }

  enum class RoundingMode {
    rte,
    rtn,
    rtz,
    rtp
  };

  struct FxpConvertFunction {
    bool Valid;
   
    RoundingMode RoundMode;
    // TODO: saturation
    bool Saturation;

    std::unique_ptr<TypeBase> SrcTy, DestTy;

    FxpConvertFunction(const std::string& input);
    
    friend std::ostream& operator<<(std::ostream& OS, 
                                    const FxpConvertFunction& Func);
  };
  

  // External math routines(e.g. sin, cos, exp)
  inline 
  bool IsFxpMathFunction(const std::string& input, 
                         typename utils::CXXABIParser::result_type* result_ptr = nullptr){
    utils::CXXABIParser parser(input);
    const auto& segs = parser.parse();
    if(result_ptr) *result_ptr = segs;

    // Currently all fixed-point related routines
    // have at leat one argument being fixed-point type
    if(segs.size() < 2 || 
       std::none_of(segs.begin(), segs.end(), 
                    [](const std::string& seg_str) {
                      return seg_str.find("fxp_") == 0;
                    })) {
      return false;
    }
   
    // TODO: Finish the list
    const auto& func_name = segs.at(0);
    return (func_name == "exp" ||
            func_name == "exp10" ||
            func_name == "log" ||
            func_name == "log10");
  }

  struct FxpMathFunction {
    bool Valid;
    std::string Name;

    // Unfortunetly, we can't use unique_ptr here
    // since none of the following options fit:
    // 1) unique_ptr<TypeBase[]>: This would allocate
    // a sequence of TypeBase memory. However, TypeBase
    // here is only used for abstracting underlying implementation, 
    // rather than a concrete object. In another word,
    // TypeBase[] would be an array of fixed-size elements,
    // but argument types(i.e. IntType, FxpType .etc) are actually 
    // vary in size.
    // 2) std::vector<unique_ptr<TypeBase>>: Element type
    // for vector should be copyable, where std::unique_ptr is not.
    // 3) unique_ptr<TypeBase>[]: I don't think it would a good idea
    // since this also should be dynamic allocated -- who gonna
    // manage it?
    std::vector<TypeBase*> ArgTypes;

    FxpMathFunction(const std::string& input);

    static std::unique_ptr<FxpMathFunction>
    Create(const std::string& input){
      std::unique_ptr<FxpMathFunction> math_func;
      math_func.reset(new FxpMathFunction(input));
      
      if(!math_func->Valid) math_func.reset(nullptr);
      return std::move(math_func);
    }

    ~FxpMathFunction();
  };
} // namespace spirv

