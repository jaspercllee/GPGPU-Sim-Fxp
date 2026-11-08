ROOT := $(abspath ../..)
CXX ?= g++
CXXFLAGS := -std=c++11 -O3 -Wall -Wno-unused-function \
	-isystem $(ROOT)/third_party/boost/include \
	-I$(ROOT)/src/cuda-sim

.PHONY: all clean
all: eval_fxp_cnn

eval_fxp_cnn: eval_fxp_cnn.cc ../mnist_mlp/fxp_requant.h $(ROOT)/src/cuda-sim/spirv_fxp_arith.cc
	$(CXX) $(CXXFLAGS) eval_fxp_cnn.cc $(ROOT)/src/cuda-sim/spirv_fxp_arith.cc -o eval_fxp_cnn

clean:
	rm -f eval_fxp_cnn
