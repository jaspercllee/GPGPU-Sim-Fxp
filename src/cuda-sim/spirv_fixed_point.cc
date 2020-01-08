#include "spirv.h"
#include <cstdlib>

#include <boost/spirit/include/phoenix.hpp>

namespace spirv {

  using namespace boost::spirit;
  namespace phx = boost::phoenix;
  static inline int initExponent(unsigned tmp_exp, unsigned meta){
    return (meta & FXP_META_MASK_EXP_NEGATIVE)? 
            -1 * (int)tmp_exp : (int)tmp_exp;
  }

#define FXP_PARSER_SUB_RULE(W, E, M) \
  ("fxp_" \
  >> uint_[phx::ref(W) = qi::_1] >> '_'  \
  >> uint_[phx::ref(E) = qi::_1] >> '_' \
  >> uint_[phx::ref(M) = qi::_1] >> '_' )

  FxpBinaryFunction::FxpBinaryFunction(const std::string& input) {
    // Don't put the following code in the constructor 
    // initialization list!
    // There would be strange behavior for OpName assigning
    Valid = qi::parse(input.begin(), input.end(),
                      "_Z" >> uint_ >>
                      "fxp_"
                      >> (qi::string("add") | qi::string("sub") | 
                          qi::string("mul") | qi::string("div"))
                          [phx::ref(OpName) = qi::_1]
                      >> uint_ 
                      >> FXP_PARSER_SUB_RULE(LhsTotalBits, lhs_exponent_, LhsMetadata)
                      >> uint_
                      >> FXP_PARSER_SUB_RULE(RhsTotalBits, rhs_exponent_, RhsMetadata)
                      );  

    LhsExponent = Valid? initExponent(lhs_exponent_, LhsMetadata) : 0;
    RhsExponent = Valid? initExponent(rhs_exponent_, RhsMetadata) : 0;
  }

  FxpConvertFunction::FxpConvertFunction(const std::string& input) {
    // DON"T put the following two lines into initialization list
    // or 'input' would be cleared!
    // Maybe we run out of stack due to potential large memory consuming
    // by qi::parse?
    RoundMode = RoundingMode::rte;
    Saturation = false;

    std::string Rounding;

    char SrcTypeCh = '#';
    std::string DestTypeStr;
    unsigned SrcFxpWidth = 0, SrcFxpMeta, src_fxp_exponent_,
             DestFxpWidth = 0, DestFxpMeta, dest_fxp_exponent_;

    Valid = qi::parse(input.begin(), input.end(),
                      "_Z" >> uint_ 
                      >> "convert_"
                      >> (
                          FXP_PARSER_SUB_RULE(DestFxpWidth, dest_fxp_exponent_, DestFxpMeta) |
                          (qi::string("int") | qi::string("uint") |
                           qi::string("short") | qi::string("ushort") |
                           qi::string("char") | qi::string("uchar") |
                           qi::string("float") | qi::string("double"))
                          [phx::ref(DestTypeStr) = qi::_1]
                         )
                      >> (qi::string("_rte") | qi::string("_rtn") | 
                          qi::string("_rtz") | qi::string("_rtp"))
                          [phx::ref(Rounding) = qi::_1]
                      >> (// DO NOT change the order of FXP_PARSER_SUB_RULE
                          // with rest of the rules
                          FXP_PARSER_SUB_RULE(SrcFxpWidth, src_fxp_exponent_, SrcFxpMeta) |
                          (qi::char_('c') | qi::char_('h') |
                          qi::char_('t') | qi::char_('s') |
                          qi::char_('i') | qi::char_('j') |
                          qi::char_('m') | qi::char_('l') |
                          qi::char_('f') | qi::char_('d'))
                          [phx::ref(SrcTypeCh) = qi::_1]
                         )
                      );

    if(Valid){
      if(Rounding == "_rte")
        RoundMode = RoundingMode::rte;
      else if(Rounding == "_rtn")
        RoundMode = RoundingMode::rtn;
      else if(Rounding == "_rtz")
        RoundMode = RoundingMode::rtz;
      else if(Rounding == "_rtp")
        RoundMode = RoundingMode::rtp;
      else
        assert(false && "Unrecognized rounding mode");
      // TODO: Saturation

      // Setup source type
      switch(SrcTypeCh){
        // 32-bits integer
        case 'j':
        case 'i': {
          auto* IT = new IntType();
          SrcTy.reset(dyn_cast<TypeBase>(IT));
          break;
        }
        // 16-bits integer
        case 's':
        case 't': {
          auto* IT = new Int16Type();
          SrcTy.reset(dyn_cast<TypeBase>(IT));
          break;
        }
        // 8-bits integer
        case 'h':
        case 'c': {
          auto* IT = new Int8Type();
          SrcTy.reset(dyn_cast<TypeBase>(IT));
          break;
        }
        case 'f': {
          auto* FT = new FloatType();
          SrcTy.reset(dyn_cast<TypeBase>(FT));
          break;
        }
        // Fixed point
        default: {
          assert(SrcFxpWidth && 
                 "Unrecognized type or incorrect fixed point width");
          int exponent = initExponent(src_fxp_exponent_, SrcFxpMeta);
          auto* fxpType = new FxpType(SrcFxpWidth, exponent, SrcFxpMeta);
          SrcTy.reset(dyn_cast<TypeBase>(fxpType));
        }
      }
      
      // Setup destination type
      if(DestTypeStr == "int" || DestTypeStr == "uint"){
        // 32-bits integer
        auto* IT = new IntType();
        DestTy.reset(dyn_cast<TypeBase>(IT));
      }else if(DestTypeStr == "short" || DestTypeStr == "ushort"){
        // 16-bits integer
        auto* IT = new Int16Type();
        DestTy.reset(dyn_cast<TypeBase>(IT));
      }else if(DestTypeStr == "char" || DestTypeStr == "uchar"){
        // 8-bits integer
        auto* IT = new Int8Type();
        DestTy.reset(dyn_cast<TypeBase>(IT));
      }else if(DestTypeStr == "float" || DestTypeStr == "double") {
        // floating number
        auto* IT = new FloatType();
        DestTy.reset(dyn_cast<TypeBase>(IT));
      }else{
        assert(DestFxpWidth && 
               "Unrecognized type or incorrect fixed point width");
        // Fixed point
        int exponent = initExponent(dest_fxp_exponent_, DestFxpMeta);
        auto* fxpType = new FxpType(DestFxpWidth, exponent, DestFxpMeta);
        DestTy.reset(dyn_cast<TypeBase>(fxpType));
      }
    }
  }

  FxpMathFunction::FxpMathFunction(const std::string& input){
    typename utils::CXXABIParser::result_type parsed_result;
    Valid = IsFxpMathFunction(input, &parsed_result);

    auto parseFxpType = [](const std::string& name) -> FxpType*{
      if(name.find("fxp_") != 0) return nullptr;

      unsigned width, raw_exp, meta;

      std::stringstream ss;
      unsigned seg_idx = 0;
      for(auto ch : name.substr(4)){
        if(ch != '_') ss << ch;
        else if(!ss.str().empty()){
          unsigned result = 0;
          ss >> result;
          switch(seg_idx++){
            case 0: // width
              width = result;
              break;
            case 1: // raw exponent
              raw_exp = result;
              break;
            case 2: // meta
              meta = result;
              break;
            default:
              assert(false && "Too many fxp template arguments");
          }
          ss.clear();
        }
      }
      return FxpType::Create(width, raw_exp, meta);
    };

    if(Valid){
      Name = parsed_result.at(0);

      // Parse argument types
      unsigned i;
      for(i = 1; i < parsed_result.size(); ++i){
        const auto& raw_arg_str = parsed_result.at(i);
        TypeBase* type_ptr = nullptr;
        if(raw_arg_str.size() == 1){
          // Primitive types
          switch(raw_arg_str.at(0)){
            // 32-bits integer
            case 'j':
            case 'i': {
              type_ptr = dyn_cast<TypeBase>(new IntType());
              break;
            }
            // 16-bits integer
            case 's':
            case 't': {
              type_ptr = dyn_cast<TypeBase>(new Int16Type());
              break;
            }
            // 8-bits integer
            case 'h':
            case 'c': {
              type_ptr = dyn_cast<TypeBase>(new Int8Type());
              break;
            }
            case 'f': {
              type_ptr = dyn_cast<TypeBase>(new FloatType());
              break;
            }
            default:
              assert(false && "Primitive argument type not supported");
          }
        }else if(raw_arg_str.find("fxp_") == 0){
          auto* fxp_type = parseFxpType(raw_arg_str);
          type_ptr = dyn_cast<TypeBase>(fxp_type);
        }else
          assert(false && "Argument type not supported");

        ArgTypes.push_back(type_ptr);
      }
    }
  }

  FxpMathFunction::~FxpMathFunction(){
    for(TypeBase* type_ptr : ArgTypes){
      if(type_ptr) delete type_ptr;
    }
  }

} // namespace spirv
