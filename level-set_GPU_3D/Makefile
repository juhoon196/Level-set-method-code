# Makefile for G-equation Level-Set Solver
#
# Usage:
#   make          - Build the solver
#   make debug    - Build with debug flags
#   make clean    - Remove build artifacts
#   make run      - Run the solver with default parameters
#   make test     - Run pyramid advection test
#   make help     - Show this help

#=============================================================================
# Compiler Settings
#=============================================================================

NVCC = nvcc
CXX = g++

# CUDA architecture (adjust for your GPU)
# Common architectures:
#   sm_60 - Pascal (GTX 1080, P100)
#   sm_70 - Volta (V100)
#   sm_75 - Turing (RTX 2080)
#   sm_80 - Ampere (A100, RTX 3080)
#   sm_86 - Ampere (RTX 3090)
#   sm_89 - Ada Lovelace (RTX 4090)
CUDA_ARCH ?= sm_70

# Compiler flags
NVCC_FLAGS = -arch=sm_89 -O3 -std=c++14
NVCC_FLAGS += -Xcompiler -Wall
NVCC_FLAGS += --use_fast_math
NVCC_FLAGS += -lineinfo

# Debug flags
DEBUG_FLAGS = -g -G -DDEBUG

# Include directories
INCLUDES = -I./include

# Library flags
LDFLAGS = -lm

#=============================================================================
# Directory Structure
#=============================================================================

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
OUTPUT_DIR = output

#=============================================================================
# Source Files
#=============================================================================

HEADERS = $(wildcard $(INC_DIR)/*.cuh)
# Only compile main.cu which includes solver.cu
SOURCES = $(SRC_DIR)/main.cu

#=============================================================================
# Output Executable
#=============================================================================

TARGET = g_equation_solver

#=============================================================================
# Build Rules
#=============================================================================

.PHONY: all clean debug run test help dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(OUTPUT_DIR)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: clean dirs $(TARGET)
	@echo "Debug build complete"

#=============================================================================
# Clean
#=============================================================================

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)
	rm -f $(OUTPUT_DIR)/*.bin $(OUTPUT_DIR)/*.vtk
	@echo "Clean complete"

distclean: clean
	rm -rf $(OUTPUT_DIR)

#=============================================================================
# Run Targets
#=============================================================================

run: $(TARGET)
	./$(TARGET)

# Test target (runs 3D deformation test)
test: $(TARGET)
	./$(TARGET)

#=============================================================================
# Help
#=============================================================================

help:
	@echo "G-equation Level-Set Solver - Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Build targets:"
	@echo "  all        - Build the solver (default)"
	@echo "  debug      - Build with debug flags"
	@echo "  clean      - Remove build artifacts"
	@echo "  distclean  - Remove all generated files"
	@echo ""
	@echo "Run targets:"
	@echo "  run        - Run solver with default options"
	@echo "  test       - Run pyramid advection test"
	@echo "  test-circle    - Run circle advection test"
	@echo "  test-no-reinit - Test without reinitialization"
	@echo "  test-quick     - Quick test (T=0.1)"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_ARCH  - CUDA architecture (default: $(CUDA_ARCH))"
	@echo ""
	@echo "Examples:"
	@echo "  make CUDA_ARCH=sm_80      - Build for Ampere GPU"
	@echo "  make test                 - Run default test"
	@echo "  ./g_equation_solver -h    - Show solver options"

#=============================================================================
# Dependency Tracking
#=============================================================================

# Force rebuild if headers change
$(TARGET): $(HEADERS)
