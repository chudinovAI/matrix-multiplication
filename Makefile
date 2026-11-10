CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3
CLANGXX ?= xcrun clang++
MACFLAGS ?= -O3 -std=c++17 -mmacosx-version-min=15.0

.DEFAULT_GOAL := all
.PHONY: all cuda metal cpu

ifeq ($(shell uname -s),Darwin)
all: metal cpu
else
all: cuda
endif

cuda: src/matmul.cu src/matmul_custom.cuh
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -shared -Xcompiler=-fPIC \
		-Xlinker=-rpath -Xlinker=$(abspath $(CUDA_HOME)/lib64) \
		-L$(CUDA_HOME)/lib64 -o build/matmul.so src/matmul.cu -lcublas

metal: src/metal_matmul.mm src/matmul.metal
	mkdir -p build
	$(CLANGXX) $(MACFLAGS) -fobjc-arc -dynamiclib src/metal_matmul.mm \
		-framework Foundation -framework Metal -framework MetalPerformanceShaders \
		-o build/metal_matmul.dylib

cpu: src/cpumat.cpp
	mkdir -p build
	$(CLANGXX) $(MACFLAGS) -DACCELERATE_NEW_LAPACK -pthread -dynamiclib src/cpumat.cpp \
		-framework Accelerate -o build/cpumat.dylib
