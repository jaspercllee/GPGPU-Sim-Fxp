#pragma once
#include "spirv.h"

void gpgpusim_spirv_invokeFxpBinaryFunction(const ptx_instruction*, 
                                            ptx_thread_info*,const function_info*, 
                                            const spirv::FxpBinaryFunction&);
void gpgpusim_spirv_invokeFxpConvertFunction(const ptx_instruction*,
                                             ptx_thread_info*, const function_info*,
                                             const spirv::FxpConvertFunction&);
void gpgpusim_spirv_invokeFxpMathFunction(const ptx_instruction*,
                                          ptx_thread_info*, const function_info*,
                                          const spirv::FxpMathFunction&);
