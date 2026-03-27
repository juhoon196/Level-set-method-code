/**
 * @file main.cu
 * @brief Main entry point for G-equation Level-Set Solver
 *
 * This program solves the G-equation using the level-set method with:
 * - WENO-5 spatial discretization
 * - TVD RK3 time integration
 * - HCR-2 reinitialization (optional)
 *
 * Usage:
 *   ./g_equation_solver [options]
 *
 * Options:
 *   -t <test>     Test case: pyramid (default), circle
 *   -n <N>        Grid size (NxN), default 201
 *   -T <time>     Final time, default 1.0
 *   -cfl <val>    CFL number, default 0.5
 *   -sl <val>     Laminar flame speed, default 0.0
 *   -u <val>      X-velocity, default 1.0
 *   -v <val>      Y-velocity, default 0.0
 *   -reinit       Enable reinitialization
 *   -no-reinit    Disable reinitialization
 *   -ri <N>       Reinit interval (steps), default 10
 *   -riter <N>    Reinit iterations, default 5
 *   -o <dir>      Output directory, default ./output
 *   -h            Show this help
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

// Include solver implementation directly to avoid multiple definition issues
// with CUDA __global__ functions across compilation units
#include "solver.cu"

//=============================================================================
// Command Line Parsing
//=============================================================================

struct CommandLineArgs {
    int test_case;          // 0: pyramid, 1: circle
    int grid_size;          // NxN grid
    double t_final;         // Final simulation time
    double cfl;             // CFL number
    double s_l;             // Laminar flame speed
    double u_vel;           // X-velocity
    double v_vel;           // Y-velocity
    bool enable_reinit;     // Reinitialization flag
    int reinit_interval;    // Reinit every N steps
    int reinit_iterations;  // Reinit iterations
    char output_dir[256];   // Output directory
    bool save_output;       // Save output files
    bool verbose;           // Verbose output
};

void printUsage(const char* prog_name) {
    printf("Usage: %s [options]\n\n", prog_name);
    printf("G-equation Level-Set Solver with WENO-5 and HCR-2 Reinitialization\n\n");
    printf("Options:\n");
    printf("  -t <test>      Test case: pyramid (default), circle, zalesak\n");
    printf("  -n <N>         Grid size (NxN), default %d\n", NX);
    printf("  -T <time>      Final time, default %.1f\n", T_FINAL);
    printf("  -cfl <val>     CFL number, default %.2f\n", CFL);
    printf("  -sl <val>      Laminar flame speed, default %.1f\n", S_L);
    printf("  -u <val>       X-velocity, default %.1f\n", U_CONST);
    printf("  -v <val>       Y-velocity, default %.1f\n", V_CONST);
    printf("  -reinit        Enable reinitialization (default if S_L > 0)\n");
    printf("  -no-reinit     Disable reinitialization\n");
    printf("  -ri <N>        Reinit interval (steps), default %d\n", REINIT_INTERVAL);
    printf("  -riter <N>     Reinit iterations, default %d\n", REINIT_ITERATIONS);
    printf("  -o <dir>       Output directory, default ./output\n");
    printf("  -no-output     Disable file output\n");
    printf("  -q             Quiet mode (less output)\n");
    printf("  -h             Show this help\n");
    printf("\nExample:\n");
    printf("  %s -t pyramid -n 201 -T 1.0 -reinit\n", prog_name);
}

CommandLineArgs parseCommandLine(int argc, char* argv[]) {
    CommandLineArgs args;

    // Set defaults
    args.test_case = 0;             // pyramid
    args.grid_size = NX;
    args.t_final = T_FINAL;
    args.cfl = CFL;
    args.s_l = S_L;
    args.u_vel = U_CONST;
    args.v_vel = V_CONST;
    args.enable_reinit = ENABLE_REINIT;
    args.reinit_interval = REINIT_INTERVAL;
    args.reinit_iterations = REINIT_ITERATIONS;
    strcpy(args.output_dir, "./output");
    args.save_output = true;
    args.verbose = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "circle") == 0) {
                args.test_case = 1;
            } else if (strcmp(argv[i], "zalesak") == 0) {
                args.test_case = 2;
            } else {
                args.test_case = 0;  // pyramid
            }
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            args.grid_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
            args.t_final = atof(argv[++i]);
        } else if (strcmp(argv[i], "-cfl") == 0 && i + 1 < argc) {
            args.cfl = atof(argv[++i]);
        } else if (strcmp(argv[i], "-sl") == 0 && i + 1 < argc) {
            args.s_l = atof(argv[++i]);
        } else if (strcmp(argv[i], "-u") == 0 && i + 1 < argc) {
            args.u_vel = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            args.v_vel = atof(argv[++i]);
        } else if (strcmp(argv[i], "-reinit") == 0) {
            args.enable_reinit = true;
        } else if (strcmp(argv[i], "-no-reinit") == 0) {
            args.enable_reinit = false;
        } else if (strcmp(argv[i], "-ri") == 0 && i + 1 < argc) {
            args.reinit_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-riter") == 0 && i + 1 < argc) {
            args.reinit_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            strcpy(args.output_dir, argv[++i]);
        } else if (strcmp(argv[i], "-no-output") == 0) {
            args.save_output = false;
        } else if (strcmp(argv[i], "-q") == 0) {
            args.verbose = false;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            exit(1);
        }
    }

    return args;
}

//=============================================================================
// CUDA Device Information
//=============================================================================

void printCUDAInfo() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("CUDA Device: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("\n");
}

//=============================================================================
// Main Simulation Function (with custom parameters)
//=============================================================================

void runCustomSimulation(const CommandLineArgs& args) {
    // Note: This is a simplified version. For full parameter customization,
    // the config.cuh parameters would need to be made runtime-configurable.
    // Currently, we use the compile-time defaults from config.cuh.

    printf("========================================\n");
    printf("G-Equation Level-Set Solver\n");
    printf("========================================\n\n");

    printCUDAInfo();

    // Run simulation with specified test case
    runSimulation(args.test_case, args.save_output);
}

//=============================================================================
// Standalone Main Function (Alternative Implementation)
//=============================================================================

/**
 * @brief Alternative main implementation that directly uses all modules
 *
 * This provides more flexibility for testing and development.
 */
void runStandaloneSimulation(const CommandLineArgs& args) {
    printf("========================================\n");
    printf("G-Equation Level-Set Solver (Standalone)\n");
    printf("========================================\n\n");

    printCUDAInfo();

    // Open log file
    FILE* log_fp = fopen("log/simulation.log", "w");
    if (log_fp) {
        fprintf(log_fp, "========================================\n");
        fprintf(log_fp, "G-Equation Level-Set Solver (Standalone)\n");
        fprintf(log_fp, "========================================\n\n");
        
        // Log device info (simplified)
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        fprintf(log_fp, "CUDA Device: %s\n\n", prop.name);
    }

    // Initialize parameters
    SimParams params = initDefaultParams();

    // Override with command line arguments
    params.s_l = args.s_l;
    params.u_const = args.u_vel;
    params.v_const = args.v_vel;
    params.t_final = args.t_final;
    params.cfl = args.cfl;
    params.enable_reinit = args.enable_reinit;
    params.reinit_interval = args.reinit_interval;
    params.reinit_iterations = args.reinit_iterations;

    // Calculate time step
    params.dt = calculateTimeStep(params);

    printParameters(params);

    if (log_fp) {
        fprintf(log_fp, "Simulation Parameters:\n");
        fprintf(log_fp, "  Grid size: %d x %d\n", params.nx, params.ny);
        fprintf(log_fp, "  dx: %.6e, dy: %.6e\n", params.dx, params.dy);
        fprintf(log_fp, "  dt: %.6e\n", params.dt);
        fprintf(log_fp, "  Final time: %.4f\n", params.t_final);
        fprintf(log_fp, "  CFL: %.2f\n", params.cfl);
        fprintf(log_fp, "  Velocity: u=%.2f, v=%.2f\n", params.u_const, params.v_const);
        fprintf(log_fp, "  Flame speed (S_L): %.2f\n", params.s_l);
        fprintf(log_fp, "  Reinitialization: %s\n", params.enable_reinit ? "Enabled" : "Disabled");
        if (params.enable_reinit) {
            fprintf(log_fp, "    Interval: %d steps\n", params.reinit_interval);
            fprintf(log_fp, "    Iterations: %d\n", params.reinit_iterations);
        }
        fprintf(log_fp, "\n");
    }

    // Allocate memory
    int total_size = params.nx_total * params.ny_total;
    size_t bytes = total_size * sizeof(double);

    printf("Allocating memory...\n");

    double* d_G, *d_G_new, *d_G_1, *d_G_2, *d_G_rhs;
    double* d_G_initial, *d_u, *d_v, *d_temp;
    double* d_G_reinit_temp, *d_G_reinit_0, *d_r_tilde;
    int* d_interface_flag;

    CUDA_CHECK(cudaMalloc(&d_G, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_1, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_2, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_rhs, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_initial, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_reinit_temp, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_reinit_0, bytes));
    CUDA_CHECK(cudaMalloc(&d_r_tilde, bytes));
    CUDA_CHECK(cudaMalloc(&d_interface_flag, total_size * sizeof(int)));

    double* h_G = new double[total_size];

    // Initialize arrays to zero
    CUDA_CHECK(cudaMemset(d_G, 0, bytes));
    CUDA_CHECK(cudaMemset(d_G_new, 0, bytes));
    CUDA_CHECK(cudaMemset(d_G_1, 0, bytes));
    CUDA_CHECK(cudaMemset(d_G_2, 0, bytes));
    CUDA_CHECK(cudaMemset(d_G_rhs, 0, bytes));

    // Initialize test case
    printf("Initializing test case...\n");
    if (args.test_case == 0) {
        initPyramidTestCase(d_G, d_u, d_v, params);
        printf("Test case: Pyramid (Diamond) Advection\n");
        if (log_fp) fprintf(log_fp, "Test case: Pyramid (Diamond) Advection\n");
    } else if (args.test_case == 1) {
        initCircleTestCase(d_G, d_u, d_v, params);
        printf("Test case: Circle Advection\n");
        if (log_fp) fprintf(log_fp, "Test case: Circle Advection\n");
    } else if (args.test_case == 2) {
        initZalesakTestCase(d_G, d_u, d_v, params);
        printf("Test case: Zalesak's Slotted Disk Rotation\n");
        if (log_fp) fprintf(log_fp, "Test case: Zalesak's Slotted Disk Rotation\n");
    }

    // Apply boundary conditions
    applyBoundaryConditions(d_G, params);
    applyVelocityBoundaryConditions(d_u, d_v, params);

    // Store initial field
    CUDA_CHECK(cudaMemcpy(d_G_initial, d_G, bytes, cudaMemcpyDeviceToDevice));

    // Calculate initial area
    double initial_area = computeInterfaceArea(d_G, d_temp, params);
    printf("Initial interface area: %.6f\n\n", initial_area);
    if (log_fp) fprintf(log_fp, "Initial interface area: %.6f\n\n", initial_area);

    // Save initial field
    if (args.save_output) {
        CUDA_CHECK(cudaMemcpy(h_G, d_G, bytes, cudaMemcpyDeviceToHost));
        saveFieldBinary("output/G_initial.bin", h_G, params);
        // saveFieldVTK("output/G_initial.vtk", h_G, params);  // VTK disabled
        printf("Saved initial field to output/G_initial.bin\n");
    }

    // Main time loop
    printf("\nStarting simulation...\n");
    printf("Target time: %.4f, dt = %.6e\n\n", params.t_final, params.dt);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventRecord(cuda_start);

    double current_time = 0.0;
    int step = 0;
    int max_steps = (int)(params.t_final / params.dt) + 100;

    while (current_time < params.t_final && step < max_steps) {
        // Adjust final step
        if (current_time + params.dt > params.t_final) {
            params.dt = params.t_final - current_time;
        }

        // RK3 time step
        rk3TimeStep(d_G, d_G_new, d_G_1, d_G_2, d_G_rhs,
                    d_u, d_v, params, applyBoundaryConditions);

        // Swap pointers
        double* temp = d_G;
        d_G = d_G_new;
        d_G_new = temp;

        // Reinitialization
        if (params.enable_reinit && step > 0 && step % params.reinit_interval == 0) {
            reinitializeWithSwap(&d_G, &d_G_reinit_temp, d_G_reinit_0,
                                  d_r_tilde, d_interface_flag,
                                  params, applyBoundaryConditions);
        }

        current_time += params.dt;
        step++;

        // Progress output and save snapshots
        if (step % OUTPUT_INTERVAL == 0) {
            double l2_error = computeL2Error(d_G, d_G_initial, d_temp, params);
            double area = computeInterfaceArea(d_G, d_temp, params);

            if (args.verbose) {
                printProgress(step, current_time, params.dt, l2_error, area);
            }

            // Save snapshot at each output interval
            if (args.save_output) {
                CUDA_CHECK(cudaMemcpy(h_G, d_G, bytes, cudaMemcpyDeviceToHost));
                char snapshot_filename[256];
                snprintf(snapshot_filename, sizeof(snapshot_filename),
                         "%s/G_step_%06d.bin", args.output_dir, step);
                saveFieldBinary(snapshot_filename, h_G, params);
            }
        }
    }

    // Stop timing
    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);
    auto end_time = std::chrono::high_resolution_clock::now();

    float cuda_elapsed_ms = 0.0f;
    cudaEventElapsedTime(&cuda_elapsed_ms, cuda_start, cuda_stop);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    auto wall_elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Final results
    printf("\n");
    double final_l2_error = computeL2Error(d_G, d_G_initial, d_temp, params);
    double final_area = computeInterfaceArea(d_G, d_temp, params);
    double shape_error = computeMeanShapeError(d_G, d_G_initial, d_temp, params);

    printSummary(final_l2_error, shape_error, initial_area, final_area);

    // Print timing results
    printf("\n========================================\n");
    printf("Timing Results:\n");
    printf("  Total steps: %d\n", step);
    printf("  GPU time: %.3f ms (%.3f s)\n", cuda_elapsed_ms, cuda_elapsed_ms / 1000.0);
    printf("  Wall time: %.3f s\n", wall_elapsed);
    printf("  Time per step: %.3f ms\n", cuda_elapsed_ms / step);
    printf("========================================\n");

    if (log_fp) {
        fprintf(log_fp, "\nFinal Results:\n");
        fprintf(log_fp, "  L2 Error: %.6e\n", final_l2_error);
        fprintf(log_fp, "  Shape Error: %.6e\n", shape_error);
        fprintf(log_fp, "  Initial Area: %.6f\n", initial_area);
        fprintf(log_fp, "  Final Area: %.6f\n", final_area);
        fprintf(log_fp, "  Area Change: %.6f%%\n", 100.0 * (final_area - initial_area) / initial_area);

        fprintf(log_fp, "\nTiming Results:\n");
        fprintf(log_fp, "  Total steps: %d\n", step);
        fprintf(log_fp, "  GPU time: %.3f ms (%.3f s)\n", cuda_elapsed_ms, cuda_elapsed_ms / 1000.0);
        fprintf(log_fp, "  Wall time: %.3f s\n", wall_elapsed);
        fprintf(log_fp, "  Time per step: %.3f ms\n", cuda_elapsed_ms / step);
        fprintf(log_fp, "========================================\n");
        fclose(log_fp);
    }

    // Save final field
    if (args.save_output) {
        CUDA_CHECK(cudaMemcpy(h_G, d_G, bytes, cudaMemcpyDeviceToHost));
        saveFieldBinary("output/G_final.bin", h_G, params);
        // saveFieldVTK("output/G_final.vtk", h_G, params);  // VTK disabled
        printf("\nSaved final field to output/G_final.bin\n");
    }

    // Cleanup
    cudaFree(d_G);
    cudaFree(d_G_new);
    cudaFree(d_G_1);
    cudaFree(d_G_2);
    cudaFree(d_G_rhs);
    cudaFree(d_G_initial);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_temp);
    cudaFree(d_G_reinit_temp);
    cudaFree(d_G_reinit_0);
    cudaFree(d_r_tilde);
    cudaFree(d_interface_flag);
    delete[] h_G;

    printf("\nSimulation complete.\n");
}

//=============================================================================
// Main Entry Point
//=============================================================================

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CommandLineArgs args = parseCommandLine(argc, argv);

    // Create output directory if it doesn't exist
    if (args.save_output) {
        char cmd[300];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", args.output_dir);
        int result = system(cmd);
        (void)result;  // Suppress unused warning
    }

    // Create log directory
    {
        int result = system("mkdir -p log");
        (void)result;
    }

    // Run simulation
    runStandaloneSimulation(args);

    return 0;
}
