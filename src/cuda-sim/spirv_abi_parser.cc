#include <iostream>
#include <sstream>
#include "spirv_abi_parser.h"

using namespace spirv;
using namespace utils;

typename CXXABIParser::result_type
CXXABIParser::parse(){
  result_type segments;
  if(raw_abi_.size() == 0) 
    return std::move(segments);
  // Starts with ABI prefix
  if(raw_abi_.find(abi_prefix_) != 0) 
    return std::move(segments);

  std::string sub_str = raw_abi_.substr(abi_prefix_.size());

  while(sub_str.size() > 0){
    std::stringstream extractor;
    extractor << sub_str;

    unsigned seg_length;
    extractor >> seg_length >> sub_str;
    if(extractor.fail()){
      // ill-format ABI string
      segments.clear();
      break;
    }
    segments.push_back(sub_str.substr(0, seg_length));
    sub_str = sub_str.substr(seg_length);
  }

  return std::move(segments);
}
