#pragma once
#include <string>
#include <vector>

namespace spirv {
namespace utils {

class CXXABIParser {
  std::string raw_abi_;

  std::string abi_prefix_;

public:
  CXXABIParser(const std::string& raw_abi_string,
               const char* abi_prefix = "_Z") :
    raw_abi_(raw_abi_string),
    abi_prefix_(abi_prefix) {}

  CXXABIParser() : raw_abi_("") {}

  using value_type = std::string;
  using result_type = std::vector<value_type>;

  result_type parse();
};

} // end namespace utils
} // end namespace spirv
