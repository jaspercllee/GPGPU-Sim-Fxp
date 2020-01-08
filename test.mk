
GTEST_VERSION := 1.8.0

GTEST_ROOT := $(shell pwd)/third_party/gtest

$(GTEST_ROOT)/%:
	wget "https://github.com/google/googletest/archive/release-$(GTEST_VERSION).tar.gz" -O \
	third_party/gtest-$(GTEST_VERSION).tar.gz
	tar -xf third_party/gtest-$(GTEST_VERSION).tar.gz -C third_party
	mv third_party/googletest-release-$(GTEST_VERSION) $(GTEST_ROOT)
	rm third_party/gtest-$(GTEST_VERSION).tar.gz
	mkdir $(GTEST_ROOT)/build

# Build Google Test Libraries
GTEST_DIR := $(GTEST_ROOT)/googletest
GTEST_SRCS_ := $(GTEST_DIR)/src/*.cc 

GTEST_CPPFLAGS_ += -isystem $(GTEST_DIR)/include
GTEST_CXXFLAGS_ += -g -Wall -Wextra -pthread

$(GTEST_ROOT)/build/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_CPPFLAGS_) -I$(GTEST_DIR) $(GTEST_CXXFLAGS_) -c \
		$(GTEST_DIR)/src/gtest-all.cc \
		-o $@

$(GTEST_ROOT)/build/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_CPPFLAGS_) -I$(GTEST_DIR) $(GTEST_CXXFLAGS_) -c \
		$(GTEST_DIR)/src/gtest_main.cc \
		-o $@

$(GTEST_ROOT)/build/libgtest.a: $(GTEST_ROOT)/build/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_ROOT)/build/libgtest_main.a: $(GTEST_ROOT)/build/gtest-all.o $(GTEST_ROOT)/build/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

gtest: $(GTEST_ROOT)/build/libgtest.a $(GTEST_ROOT)/build/libgtest_main.a

GTEST_INC_PATH := $(GTEST_DIR)/include
GTEST_LIB_PATH := $(GTEST_ROOT)/build
