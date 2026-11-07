# Standalone fixed-point baseline. No CUDA, OpenCL, or setup_environment.
ROOT := $(abspath ../..)
CXX ?= g++
CXXFLAGS := -std=c++11 -O3 -DNDEBUG -Wall -Wno-unused-function \
	-isystem $(ROOT)/third_party/boost/include \
	-I$(ROOT)/src/cuda-sim

SRCS := \
	$(ROOT)/src/cuda-sim/spirv_fixed_point.cc \
	$(ROOT)/src/cuda-sim/spirv_abi_parser.cc \
	$(ROOT)/src/cuda-sim/spirv_verbose_utils.cc \
	$(ROOT)/src/cuda-sim/spirv_fxp_arith.cc \
	cpu_fxp_tests.cc

OUT := build/cpu_fxp_tests

.PHONY: all clean
all: $(OUT)
	./$(OUT)

$(OUT): $(SRCS) $(ROOT)/src/cuda-sim/spirv_fixed_point.h $(ROOT)/src/cuda-sim/spirv_fxp_arith.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(OUT)

clean:
	rm -rf build
