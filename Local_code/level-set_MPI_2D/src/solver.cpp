/**
 * @file solver.cpp
 * @brief G-equation Level-Set Solver implementation (MPI version)
 *
 * This file implements the main solver class that coordinates all
 * components of the level-set method:
 * - WENO-5 spatial discretization
 * - TVD RK3 time integration
 * - HCR-2 reinitialization
 * - MPI domain decomposition and communication
 */

#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "../include/config.h"
#include "../include/weno5.h"
#include "../include/rk3.h"
#include "../include/reinitialization.h"
#include "../include/boundary.h"
#include "../include/initial_conditions.h"
#include "../include/io.h"

//=============================================================================
// G-Equation Solver Class (MPI Version)
//=============================================================================

/**
 * @class GEquationSolver
 * @brief Main solver class for the G-equation level-set method
 */
class GEquationSolver {
public:
    // Simulation parameters
    SimParams params;

    // Local arrays (allocated per process)
    double* G;              // Current G field
    double* G_new;          // New G field after time step
    double* G_1;            // RK3 intermediate stage 1
    double* G_2;            // RK3 intermediate stage 2
    double* G_rhs;          // RHS for RK3
    double* G_initial;      // Initial G field (for error calculation)
    double* u;              // Velocity x-component
    double* v;              // Velocity y-component

    // Reinitialization arrays
    double* G_reinit_temp;
    double* G_reinit_0;
    double* r_tilde;
    int* interface_flag;

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
        G = nullptr;
        G_new = nullptr;
        G_1 = nullptr;
        G_2 = nullptr;
        G_rhs = nullptr;
        G_initial = nullptr;
        u = nullptr;
        v = nullptr;
        G_reinit_temp = nullptr;
        G_reinit_0 = nullptr;
        r_tilde = nullptr;
        interface_flag = nullptr;
    }

    /**
     * @brief Destructor - clean up all allocations
     */
    ~GEquationSolver() {
        cleanup();
    }

    /**
     * @brief Setup MPI and domain decomposition
     */
    void setupMPI(int rank, int num_procs) {
        setupMPIDomainDecomposition(params, rank, num_procs);
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
     * @brief Allocate all local memory
     */
    bool allocateMemory() {
        int local_total_size = params.nx_total * params.local_ny_total;
        size_t bytes = local_total_size * sizeof(double);

        if (params.rank == 0) {
            printf("Allocating memory: %d x %d = %d cells per process (%.2f MB)\n",
                   params.nx_total, params.local_ny_total, local_total_size,
                   bytes * 12 / (1024.0 * 1024.0));
        }

        // Allocate local arrays
        G = new double[local_total_size]();
        G_new = new double[local_total_size]();
        G_1 = new double[local_total_size]();
        G_2 = new double[local_total_size]();
        G_rhs = new double[local_total_size]();
        G_initial = new double[local_total_size]();
        u = new double[local_total_size]();
        v = new double[local_total_size]();

        // Reinitialization arrays
        G_reinit_temp = new double[local_total_size]();
        G_reinit_0 = new double[local_total_size]();
        r_tilde = new double[local_total_size]();
        interface_flag = new int[local_total_size]();

        if (params.rank == 0) {
            printf("Memory allocation complete.\n");
        }

        return true;
    }

    /**
     * @brief Free all allocated memory
     */
    void cleanup() {
        delete[] G; G = nullptr;
        delete[] G_new; G_new = nullptr;
        delete[] G_1; G_1 = nullptr;
        delete[] G_2; G_2 = nullptr;
        delete[] G_rhs; G_rhs = nullptr;
        delete[] G_initial; G_initial = nullptr;
        delete[] u; u = nullptr;
        delete[] v; v = nullptr;
        delete[] G_reinit_temp; G_reinit_temp = nullptr;
        delete[] G_reinit_0; G_reinit_0 = nullptr;
        delete[] r_tilde; r_tilde = nullptr;
        delete[] interface_flag; interface_flag = nullptr;

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
        initPyramidTestCase(G, u, v, params);

        // Apply boundary conditions
        applyBoundaryConditions(G, params);
        applyVelocityBoundaryConditions(u, v, params);

        // Store initial field for error calculation
        copyInitialField(G, G_initial, params.nx_total * params.local_ny_total);

        // Reset simulation state
        current_time = 0.0;
        current_step = 0;
        is_initialized = true;

        printParameters(params);
        if (params.rank == 0) {
            printf("Initialized Pyramid Advection Test Case\n\n");
        }

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

        initCircleTestCase(G, u, v, params);
        applyBoundaryConditions(G, params);
        applyVelocityBoundaryConditions(u, v, params);
        copyInitialField(G, G_initial, params.nx_total * params.local_ny_total);

        current_time = 0.0;
        current_step = 0;
        is_initialized = true;

        printParameters(params);
        if (params.rank == 0) {
            printf("Initialized Circle Advection Test Case\n\n");
        }

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

        initZalesakTestCase(G, u, v, params);
        applyBoundaryConditions(G, params);
        applyVelocityBoundaryConditions(u, v, params);
        copyInitialField(G, G_initial, params.nx_total * params.local_ny_total);

        current_time = 0.0;
        current_step = 0;
        is_initialized = true;

        printParameters(params);

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
        rk3TimeStep(G, G_new, G_1, G_2, G_rhs, u, v, params);

        // Swap pointers
        double* temp = G;
        G = G_new;
        G_new = temp;

        // Reinitialization (if enabled and at correct interval)
        if (params.enable_reinit &&
            current_step > 0 &&
            current_step % params.reinit_interval == 0) {

            reinitializeWithSwap(&G, &G_reinit_temp, G_reinit_0,
                                 r_tilde, interface_flag, params);
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

        if (params.rank == 0) {
            printf("Starting simulation...\n");
            printf("Target time: %.4f, dt = %.6e, estimated steps: %d\n\n",
                   params.t_final, params.dt, (int)(params.t_final / params.dt));
        }

        // Calculate initial metrics
        double initial_area = computeInterfaceArea(G, params);
        if (params.rank == 0) {
            printf("Initial interface area: %.6f\n\n", initial_area);
        }

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
                double l2_error = computeL2Error(G, G_initial, params);
                double area = computeInterfaceArea(G, params);
                printProgress(current_step, current_time, params.dt, l2_error, area, params.rank);
            }
        }

        // Final output
        double final_l2_error = computeL2Error(G, G_initial, params);
        double final_area = computeInterfaceArea(G, params);
        double shape_error = computeMeanShapeError(G, G_initial, params);

        printSummary(final_l2_error, shape_error, initial_area, final_area, params.rank);
    }

    /**
     * @brief Save current G field to binary file
     */
    bool saveField(const char* filename) {
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", output_dir, filename);
        return saveFieldBinary(fullpath, G, params);
    }

    /**
     * @brief Save current G field to VTK file
     */
    bool saveFieldAsVTK(const char* filename) {
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", output_dir, filename);
        return saveFieldVTK(fullpath, G, params);
    }

    /**
     * @brief Save initial G field
     */
    bool saveInitialField(const char* filename) {
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", output_dir, filename);
        return saveFieldBinary(fullpath, G_initial, params);
    }

    /**
     * @brief Get current L2 error
     */
    double getL2Error() {
        return computeL2Error(G, G_initial, params);
    }

    /**
     * @brief Get current interface area
     */
    double getInterfaceArea() {
        return computeInterfaceArea(G, params);
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
 * @brief Run full simulation with specified test case
 */
void runSimulation(int test_case, bool save_output, int rank, int num_procs) {
    GEquationSolver solver;
    solver.setupMPI(rank, num_procs);

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
            if (rank == 0) {
                fprintf(stderr, "Unknown test case: %d\n", test_case);
            }
            return;
    }

    if (!init_success) {
        if (rank == 0) {
            fprintf(stderr, "Initialization failed\n");
        }
        return;
    }

    // Save initial field
    if (save_output) {
        solver.saveField("G_initial.bin");
    }

    // Run simulation
    solver.run();

    // Save final field
    if (save_output) {
        solver.saveField("G_final.bin");
    }
}
