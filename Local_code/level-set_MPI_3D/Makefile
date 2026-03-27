# Makefile for G-equation Level-Set Solver 3D (MPI Version)
#
# Usage:
#   make          - Build the solver
#   make debug    - Build with debug flags
#   make clean    - Remove build artifacts
#   make run      - Run the solver with 4 MPI processes
#   make test     - Run sphere advection test
#   make help     - Show this help

#=============================================================================
# Compiler Settings
#=============================================================================

MPICXX = mpicxx

CXXFLAGS = -O3 -std=c++14 -Wall -Wextra
CXXFLAGS += -march=native

DEBUG_FLAGS = -g -O0 -DDEBUG

INCLUDES = -I./include

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

HEADERS = $(wildcard $(INC_DIR)/*.h)
SOURCES = $(SRC_DIR)/main.cpp

#=============================================================================
# Output Executable
#=============================================================================

TARGET = g_equation_solver_3d

#=============================================================================
# Default number of MPI processes
#=============================================================================

NP ?= 4

#=============================================================================
# Build Rules
#=============================================================================

.PHONY: all clean debug run test help dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(OUTPUT_DIR)

$(TARGET): $(SOURCES) $(HEADERS)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

debug: CXXFLAGS += $(DEBUG_FLAGS)
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
	mpirun -np $(NP) ./$(TARGET)

# Sphere advection test (default)
test: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) -t sphere -T 1.0 -u 1.0 -v 0.0 -w 0.0 -no-reinit

# Sphere advection with reinit
test-reinit: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) -t sphere -T 1.0 -reinit

# Zalesak sphere test
test-zalesak: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) -t zalesak -T 6.283185 -reinit

# Quick test (shorter time)
test-quick: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) -t sphere -T 0.1 -no-reinit

# Single process test (for debugging)
test-single: $(TARGET)
	mpirun -np 1 ./$(TARGET) -t sphere -T 0.1 -no-reinit

# Scalability test
test-scale: $(TARGET)
	@echo "Testing scalability..."
	@for np in 1 2 4 8; do \
		echo "=== NP=$$np ==="; \
		mpirun -np $$np ./$(TARGET) -t sphere -T 0.5 -no-reinit -q; \
	done

#=============================================================================
# Help
#=============================================================================

help:
	@echo "G-equation Level-Set Solver 3D (MPI) - Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Build targets:"
	@echo "  all           - Build the solver (default)"
	@echo "  debug         - Build with debug flags"
	@echo "  clean         - Remove build artifacts"
	@echo "  distclean     - Remove all generated files"
	@echo ""
	@echo "Run targets:"
	@echo "  run           - Run solver with default options (NP=$(NP) processes)"
	@echo "  test          - Run sphere advection test"
	@echo "  test-reinit   - Run sphere test with reinitialization"
	@echo "  test-zalesak  - Run Zalesak slotted sphere rotation test"
	@echo "  test-quick    - Quick test (T=0.1)"
	@echo "  test-single   - Single process test (debugging)"
	@echo "  test-scale    - Scalability test with 1,2,4,8 processes"
	@echo ""
	@echo "Configuration:"
	@echo "  NP            - Number of MPI processes (default: $(NP))"
	@echo ""
	@echo "Examples:"
	@echo "  make NP=8 test      - Run test with 8 processes"
	@echo "  make test-single    - Run with 1 process for debugging"
	@echo "  ./g_equation_solver_3d -h - Show solver options"

#=============================================================================
# Dependency Tracking
#=============================================================================

$(TARGET): $(HEADERS)
