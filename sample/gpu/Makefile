# CUDA Makefile for G-equation GPU Solver

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_70 --use_fast_math
NVCC_FLAGS += -Xcompiler -fopenmp

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -g -G -lineinfo

TARGET = geqn_gpu
SOURCES = main.cu field_cuda.cu weno5_kernels.cu rk3_solver.cu
HEADERS = config_cuda.cuh field_cuda.cuh weno5_kernels.cuh rk3_solver.cuh

OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJECTS)

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

run: $(TARGET)
	./$(TARGET) 100

profile: $(TARGET)
	nvprof ./$(TARGET) 100

.PHONY: all clean run profile
