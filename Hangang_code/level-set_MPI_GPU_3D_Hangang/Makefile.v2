# Makefile for G-equation Level-Set Solver (MPI + CUDA multi-GPU)
#
# Usage:
#   make                      - Build the solver
#   make CUDA_ARCH=sm_80      - Build for Ampere
#   make run NP=4             - Run with 4 MPI ranks (4 GPUs)
#   make clean                - Remove build artifacts

#=============================================================================
# Compiler Settings
#=============================================================================

NVCC = nvcc

# CUDA architecture (adjust for your GPU)
CUDA_ARCH ?= sm_90

# Host compiler (must be compatible with nvcc — g++-9 works, g++-11 does not)
HOST_CXX ?= g++

# MPI paths (auto-detected from mpicxx, override with MPI_INCLUDE / MPI_LIB)
MPI_INCLUDE ?= -I/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/include
MPI_LIB ?= -L/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib -lmpi -lmpi_gtl_cuda

NVCC_FLAGS  = -arch=$(CUDA_ARCH) -O3 -std=c++14 -lineinfo --maxrregcount=64
NVCC_FLAGS += -ccbin $(HOST_CXX)
NVCC_FLAGS += -Xcompiler -Wall
NVCC_FLAGS += --use_fast_math
NVCC_FLAGS += -lineinfo

DEBUG_FLAGS = -g -G -DDEBUG

INCLUDES = -I./include $(MPI_INCLUDE)
LDFLAGS  = -lm $(MPI_LIB)

#=============================================================================
# Directories
#=============================================================================

SRC_DIR    = src
INC_DIR    = include
BUILD_DIR  = build
OUTPUT_DIR = output

#=============================================================================
# Sources
#=============================================================================

HEADERS = $(wildcard $(INC_DIR)/*.cuh)
SOURCES = $(SRC_DIR)/main.cu
TARGET  = g_equation_solver_mpi_gpu

#=============================================================================
# Number of MPI ranks for 'make run'
#=============================================================================

NP ?= 2

#=============================================================================
# Build Rules
#=============================================================================

.PHONY: all clean debug run dirs help

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR) $(OUTPUT_DIR) log

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: clean dirs $(TARGET)
	@echo "Debug build complete"

#=============================================================================
# Run
#=============================================================================

run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

#=============================================================================
# Clean
#=============================================================================

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)
	rm -f $(OUTPUT_DIR)/*.bin $(OUTPUT_DIR)/*.vtk
	@echo "Clean complete"

distclean: clean
	rm -rf $(OUTPUT_DIR) log

#=============================================================================
# Help
#=============================================================================

help:
	@echo "G-equation Level-Set Solver (MPI + CUDA multi-GPU)"
	@echo ""
	@echo "  make                       Build (default arch=$(CUDA_ARCH))"
	@echo "  make CUDA_ARCH=sm_80       Build for specific GPU"
	@echo "  make run NP=4              Run with 4 MPI ranks"
	@echo "  make debug                 Debug build"
	@echo "  make clean                 Remove build artefacts"

# Force rebuild when headers change
$(TARGET): $(HEADERS)
