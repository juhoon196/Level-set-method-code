/**
 * @file solver.cu
 * @brief G-equation Level-Set Solver implementation
 *
 * This file implements the main solver class that coordinates all
 * components of the level-set method:
 * - WENO-5 spatial discretization
 * - TVD RK3 time integration
 * - HCR-2 reinitialization
 * - Boundary condition handling
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#include "../include/config.cuh"
#include "../include/weno5.cuh"
#include "../include/rk3.cuh"
#include "../include/reinitialization.cuh"
#include "../include/boundary.cuh"
#include "../include/initial_conditions.cuh"
#include "../include/io.cuh"

//=============================================================================
// G-Equation Solver Class
//=============================================================================

/**
 * @class GEquationSolver
 * @brief Main solver class for the G-equation level-set method
 */
class GEquationSolver {
public:
    // Simulation parameters
    SimParams params;

    // Device arrays
    double* d_G;            // Current G field
    double* d_G_new;        // New G field after time step
    double* d_G_1;          // RK3 intermediate stage 1
    double* d_G_2;          // RK3 intermediate stage 2
    double* d_G_rhs;        // RHS for RK3
    double* d_G_initial;    // Initial G field (for error calculation)
    double* d_u;            // Velocity x-component
    double* d_v;            // Velocity y-component

    // Reinitialization arrays
    double* d_G_reinit_temp;
    double* d_G_reinit_0;
    double* d_r_tilde;
    int* d_interface_flag;

    // Temporary array for error calculation
    double* d_temp;

    // Host arrays for I/O
    double* h_G;

    // Simulation state
    double current_time;
    int current_step;
    bool is_initialized;

    // Output directory
    char output_dir[256];

    /**
     * @brief Constructor - initialize with default parameters
     */
    GEquationSolver() : is_initialized(false), current_time(0.0), current_step(0) {
        params = initDefaultParams();
        strcpy(output_dir, "./output");

        // Initialize all pointers to null
        d_G = nullptr;
        d_G_new = nullptr;
        d_G_1 = nullptr;
        d_G_2 = nullptr;
        d_G_rhs = nullptr;
        d_G_initial = nullptr;
        d_u = nullptr;
        d_v = nullptr;
        d_G_reinit_temp = nullptr;
        d_G_reinit_0 = nullptr;
        d_r_tilde = nullptr;
        d_interface_flag = nullptr;
        d_temp = nullptr;
        h_G = nullptr;
    }

    /**
     * @brief Destructor - clean up all allocations
     */
    ~GEquationSolver() {
        cleanup();
    }

    /**
     * @brief Set simulation parameters
     */
    void setParameters(const SimParams& new_params) {
        params = new_params;
    }

    /**
     * @brief Set output directory
     */
    void setOutputDirectory(const char* dir) {
        strcpy(output_dir, dir);
    }

    /**
     * @brief Allocate all device and host memory
     */
    bool allocateMemory() {
        int total_size = params.nx_total * params.ny_total;
        size_t bytes = total_size * sizeof(double);

        printf("Allocating memory: %d x %d = %d cells (%.2f MB)\n",
               params.nx_total, params.ny_total, total_size,
               bytes * 12 / (1024.0 * 1024.0));  // ~12 arrays

        // Device arrays for main solver
        CUDA_CHECK(cudaMalloc(&d_G, bytes));
        CUDA_CHECK(cudaMalloc(&d_G_new, bytes));
        CUDA_CHECK(cudaMalloc(&d_G_1, bytes));
        CUDA_CHECK(cudaMalloc(&d_G_2, bytes));
        CUDA_CHECK(cudaMalloc(&d_G_rhs, bytes));
        CUDA_CHECK(cudaMalloc(&d_G_initial, bytes));
        CUDA_CHECK(cudaMalloc(&d_u, bytes));
        CUDA_CHECK(cudaMalloc(&d_v, bytes));
        CUDA_CHECK(cudaMalloc(&d_temp, bytes));

        // Device arrays for reinitialization
        CUDA_CHECK(cudaMalloc(&d_G_reinit_temp, bytes));
        CUDA_CHECK(cudaMalloc(&d_G_reinit_0, bytes));
        CUDA_CHECK(cudaMalloc(&d_r_tilde, bytes));
        CUDA_CHECK(cudaMalloc(&d_interface_flag, total_size * sizeof(int)));

        // Host array for I/O
        h_G = new double[total_size];

        // Initialize device arrays to zero
        CUDA_CHECK(cudaMemset(d_G, 0, bytes));
        CUDA_CHECK(cudaMemset(d_G_new, 0, bytes));
        CUDA_CHECK(cudaMemset(d_G_1, 0, bytes));
        CUDA_CHECK(cudaMemset(d_G_2, 0, bytes));
        CUDA_CHECK(cudaMemset(d_G_rhs, 0, bytes));
        CUDA_CHECK(cudaMemset(d_G_initial, 0, bytes));
        CUDA_CHECK(cudaMemset(d_u, 0, bytes));
        CUDA_CHECK(cudaMemset(d_v, 0, bytes));
        CUDA_CHECK(cudaMemset(d_temp, 0, bytes));

        printf("Memory allocation complete.\n");
        return true;
    }

    /**
     * @brief Free all allocated memory
     */
    void cleanup() {
        if (d_G) cudaFree(d_G);
        if (d_G_new) cudaFree(d_G_new);
        if (d_G_1) cudaFree(d_G_1);
        if (d_G_2) cudaFree(d_G_2);
        if (d_G_rhs) cudaFree(d_G_rhs);
        if (d_G_initial) cudaFree(d_G_initial);
        if (d_u) cudaFree(d_u);
        if (d_v) cudaFree(d_v);
        if (d_temp) cudaFree(d_temp);
        if (d_G_reinit_temp) cudaFree(d_G_reinit_temp);
        if (d_G_reinit_0) cudaFree(d_G_reinit_0);
        if (d_r_tilde) cudaFree(d_r_tilde);
        if (d_interface_flag) cudaFree(d_interface_flag);
        if (h_G) delete[] h_G;

        d_G = nullptr;
        d_G_new = nullptr;
        d_G_1 = nullptr;
        d_G_2 = nullptr;
        d_G_rhs = nullptr;
        d_G_initial = nullptr;
        d_u = nullptr;
        d_v = nullptr;
        d_temp = nullptr;
        d_G_reinit_temp = nullptr;
        d_G_reinit_0 = nullptr;
        d_r_tilde = nullptr;
        d_interface_flag = nullptr;
        h_G = nullptr;

        is_initialized = false;
    }

    /**
     * @brief Initialize simulation with pyramid test case
     */
    bool initializePyramidTest() {
        if (!allocateMemory()) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            return false;
        }

        // Calculate time step
        params.dt = calculateTimeStep(params);

        // Initialize fields
        initPyramidTestCase(d_G, d_u, d_v, params);

        // Apply boundary conditions
        applyBoundaryConditions(d_G, params);
        applyVelocityBoundaryConditions(d_u, d_v, params);

        // Store initial field for error calculation
        copyInitialField(d_G, d_G_initial, params.nx_total * params.ny_total);

        // Reset simulation state
        current_time = 0.0;
        current_step = 0;
        is_initialized = true;

        printParameters(params);
        printf("Initialized Pyramid Advection Test Case\n\n");

        return true;
    }

    /**
     * @brief Initialize simulation with circle test case
     */
    bool initializeCircleTest() {
        if (!allocateMemory()) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            return false;
        }

        params.dt = calculateTimeStep(params);

        initCircleTestCase(d_G, d_u, d_v, params);
        applyBoundaryConditions(d_G, params);
        applyVelocityBoundaryConditions(d_u, d_v, params);
        copyInitialField(d_G, d_G_initial, params.nx_total * params.ny_total);

        current_time = 0.0;
        current_step = 0;
        is_initialized = true;

        printParameters(params);
        printf("Initialized Circle Advection Test Case\n\n");

        return true;
    }

    /**
     * @brief Initialize simulation with Zalesak's slotted disk test case
     */
    bool initializeZalesakTest() {
        if (!allocateMemory()) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            return false;
        }

        params.dt = calculateTimeStep(params);

        initZalesakTestCase(d_G, d_u, d_v, params);
        applyBoundaryConditions(d_G, params);
        applyVelocityBoundaryConditions(d_u, d_v, params);
        copyInitialField(d_G, d_G_initial, params.nx_total * params.ny_total);

        current_time = 0.0;
        current_step = 0;
        is_initialized = true;

        printParameters(params);
        printf("Initialized Zalesak's Slotted Disk Test Case\n\n");

        return true;
    }

    /**
     * @brief Perform a single time step
     */
    void step() {
        if (!is_initialized) {
            fprintf(stderr, "Error: Solver not initialized\n");
            return;
        }

        // RK3 time integration
        rk3TimeStep(d_G, d_G_new, d_G_1, d_G_2, d_G_rhs,
                    d_u, d_v, params, applyBoundaryConditions);

        // Swap pointers
        double* temp = d_G;
        d_G = d_G_new;
        d_G_new = temp;

        // Reinitialization (if enabled and at correct interval)
        if (params.enable_reinit &&
            current_step > 0 &&
            current_step % params.reinit_interval == 0) {

            reinitializeWithSwap(&d_G, &d_G_reinit_temp, d_G_reinit_0,
                                  d_r_tilde, d_interface_flag,
                                  params, applyBoundaryConditions);
        }

        // Update time
        current_time += params.dt;
        current_step++;
    }

    /**
     * @brief Run simulation until final time
     */
    void run() {
        if (!is_initialized) {
            fprintf(stderr, "Error: Solver not initialized\n");
            return;
        }

        printf("Starting simulation...\n");
        printf("Target time: %.4f, dt = %.6e, estimated steps: %d\n\n",
               params.t_final, params.dt, (int)(params.t_final / params.dt));

        // Calculate initial metrics
        double initial_area = computeInterfaceArea(d_G, d_temp, params);
        printf("Initial interface area: %.6f\n\n", initial_area);

        // Main time loop
        while (current_time < params.t_final && current_step < MAX_STEPS) {
            // Adjust final step to hit exact end time
            if (current_time + params.dt > params.t_final) {
                params.dt = params.t_final - current_time;
            }

            // Perform time step
            step();

            // Output progress
            if (current_step % OUTPUT_INTERVAL == 0) {
                double l2_error = computeL2Error(d_G, d_G_initial, d_temp, params);
                double area = computeInterfaceArea(d_G, d_temp, params);
                printProgress(current_step, current_time, params.dt, l2_error, area);
            }
        }

        // Final output
        printf("\n");
        double final_l2_error = computeL2Error(d_G, d_G_initial, d_temp, params);
        double final_area = computeInterfaceArea(d_G, d_temp, params);
        double shape_error = computeMeanShapeError(d_G, d_G_initial, d_temp, params);

        printSummary(final_l2_error, shape_error, initial_area, final_area);
    }

    /**
     * @brief Save current G field to binary file
     */
    bool saveField(const char* filename) {
        int total_size = params.nx_total * params.ny_total;
        CUDA_CHECK(cudaMemcpy(h_G, d_G, total_size * sizeof(double), cudaMemcpyDeviceToHost));

        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", output_dir, filename);

        return saveFieldBinary(fullpath, h_G, params);
    }

    /**
     * @brief Save current G field to VTK file
     */
    bool saveFieldAsVTK(const char* filename) {
        int total_size = params.nx_total * params.ny_total;
        CUDA_CHECK(cudaMemcpy(h_G, d_G, total_size * sizeof(double), cudaMemcpyDeviceToHost));

        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", output_dir, filename);

        return saveFieldVTK(fullpath, h_G, params);
    }

    /**
     * @brief Save initial G field
     */
    bool saveInitialField(const char* filename) {
        int total_size = params.nx_total * params.ny_total;
        CUDA_CHECK(cudaMemcpy(h_G, d_G_initial, total_size * sizeof(double), cudaMemcpyDeviceToHost));

        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", output_dir, filename);

        return saveFieldBinary(fullpath, h_G, params);
    }

    /**
     * @brief Get current L2 error
     */
    double getL2Error() {
        return computeL2Error(d_G, d_G_initial, d_temp, params);
    }

    /**
     * @brief Get current interface area
     */
    double getInterfaceArea() {
        return computeInterfaceArea(d_G, d_temp, params);
    }

    /**
     * @brief Get current time
     */
    double getCurrentTime() const {
        return current_time;
    }

    /**
     * @brief Get current step
     */
    int getCurrentStep() const {
        return current_step;
    }
};

//=============================================================================
// Convenience Functions for External Use
//=============================================================================

/**
 * @brief Create and initialize solver with pyramid test case
 */
GEquationSolver* createPyramidTestSolver() {
    GEquationSolver* solver = new GEquationSolver();

    if (!solver->initializePyramidTest()) {
        delete solver;
        return nullptr;
    }

    return solver;
}

/**
 * @brief Create and initialize solver with circle test case
 */
GEquationSolver* createCircleTestSolver() {
    GEquationSolver* solver = new GEquationSolver();

    if (!solver->initializeCircleTest()) {
        delete solver;
        return nullptr;
    }

    return solver;
}

/**
 * @brief Run full simulation with specified test case
 */
void runSimulation(int test_case, bool save_output) {
    GEquationSolver solver;

    // Initialize based on test case
    bool init_success = false;
    switch (test_case) {
        case 0:
            init_success = solver.initializePyramidTest();
            break;
        case 1:
            init_success = solver.initializeCircleTest();
            break;
        case 2:
            init_success = solver.initializeZalesakTest();
            break;
        default:
            fprintf(stderr, "Unknown test case: %d\n", test_case);
            return;
    }

    if (!init_success) {
        fprintf(stderr, "Initialization failed\n");
        return;
    }

    // Save initial field
    if (save_output) {
        solver.saveField("G_initial.bin");
        // solver.saveFieldAsVTK("G_initial.vtk");  // VTK disabled
    }

    // Run simulation
    solver.run();

    // Save final field
    if (save_output) {
        solver.saveField("G_final.bin");
        // solver.saveFieldAsVTK("G_final.vtk");  // VTK disabled
    }
}
