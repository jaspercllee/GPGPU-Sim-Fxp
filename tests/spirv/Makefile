
# Historical Google Test driver. Needs gtest 1.8 (see test.mk) and a built
# libOpenCL.so. CI runs tests/cpu_fxp instead. Multiply expectations in
# fixed_point_arithmetic_impl.cc do not match spirv_fxp_arith.cc.
OUTPUT_DIR := $(SIM_OBJ_FILES_DIR)/tests/spirv

CPP_FLAGS := -g3 -std=c++11 \
		-I$(GTEST_INC_PATH) \
		-I$(BOOST_PATH)/include \
		-I../../src/cuda-sim

LD_FLAGS := -L$(GTEST_LIB_PATH) \
		-lgtest_main -lpthread \
		-L$(OCL_LIB_PATH) \
		-lOpenCL

SRCS := $(shell ls *.cc)
OBJS := $(SRCS:%.cc=$(OUTPUT_DIR)/%.o)
OUT_EXE := $(OUTPUT_DIR)/$(EXE_NAME)

$(OUTPUT_DIR)/%.o: %.cc
	$(CXX) $(CPP_FLAGS) -c $< -o $@

$(OUT_EXE): $(OBJS)
	$(CXX) $(CPP_FLAGS) $^ -o $@ $(LD_FLAGS)

all: $(OUT_EXE)
