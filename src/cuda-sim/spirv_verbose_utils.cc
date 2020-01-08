#include "spirv_fixed_point.h"

namespace spirv {

std::ostream& operator<<(std::ostream& OS, const FxpBinaryFunction& F){
  if(F.Valid){
    OS << F.OpName << "<";
    OS << "(" << F.LhsTotalBits << ":" << F.LhsExponent << ":" << F.LhsMetadata << ")";
    OS << ",";
    OS << "(" << F.RhsTotalBits << ":" << F.RhsExponent << ":" << F.RhsMetadata << ")";
    OS << ">";
  }else
    OS << "Invalid Fxp binary function";
  return OS;
} 

} // end namespace spirv
