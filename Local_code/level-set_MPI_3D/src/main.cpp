/**
 * @file main.cpp
 * @brief Main entry point for G-equation Level-Set Solver 3D (MPI version)
 *
 * Usage:
 *   mpirun -np <N> ./g_equation_solver_3d [options]
 *
 * Options:
 *   -t <test>     Test case: sphere (default), zalesak
 *   -T <time>     Final time, default 1.5
 *   -cfl <val>    CFL number, default 0.2
 *   -sl <val>     Laminar flame speed, default 0.0
 *   -u <val>      X-velocity, default 0.0
 *   -v <val>      Y-velocity, default 0.0
 *   -w <val>      Z-velocity, default 0.0
 *   -reinit       Enable reinitialization
 *   -no-reinit    Disable reinitialization
 *   -ri <N>       Reinit interval (steps), default 10
 *   -riter <N>    Reinit iterations, default 2
 *   -o <dir>      Output directory, default ./output
 *   -h            Show this help
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../include/config.h"
#include "../include/weno5.h"
#include "../include/rk3.h"
#include "../include/reinitialization.h"
#include "../include/boundary.h"
#include "../include/initial_conditions.h"
#include "../include/io.h"

//=============================================================================
// Command Line Parsing
//=============================================================================

struct CommandLineArgs {
    int test_case;          // 0: sphere, 1: zalesak
    double t_final;
    double cfl;
    double s_l;
    double u_vel;
    double v_vel;
    double w_vel;
    bool enable_reinit;
    int reinit_interval;
    int reinit_iterations;
    char output_dir[256];
    bool save_output;
    bool verbose;
};

void printUsage(const char* prog_name, int rank) {
    if (rank != 0) return;

    printf("Usage: mpirun -np <N> %s [options]\n\n", prog_name);
    printf("G-equation Level-Set Solver 3D with WENO-5 and HCR-2 (MPI)\n\n");
    printf("Options:\n");
    printf("  -t <test>      Test case: sphere (default), zalesak\n");
    printf("  -T <time>      Final time, default %.4f\n", T_FINAL);
    printf("  -cfl <val>     CFL number, default %.2f\n", CFL);
    printf("  -sl <val>      Laminar flame speed, default %.1f\n", S_L);
    printf("  -u <val>       X-velocity, default %.1f\n", U_CONST);
    printf("  -v <val>       Y-velocity, default %.1f\n", V_CONST);
    printf("  -w <val>       Z-velocity, default %.1f\n", W_CONST);
    printf("  -reinit        Enable reinitialization\n");
    printf("  -no-reinit     Disable reinitialization\n");
    printf("  -ri <N>        Reinit interval (steps), default %d\n", REINIT_INTERVAL);
    printf("  -riter <N>     Reinit iterations, default %d\n", REINIT_ITERATIONS);
    printf("  -o <dir>       Output directory, default ./output\n");
    printf("  -no-output     Disable file output\n");
    printf("  -q             Quiet mode (less output)\n");
    printf("  -h             Show this help\n");
    printf("\nExample:\n");
    printf("  mpirun -np 4 %s -t sphere -T 1.0 -reinit\n", prog_name);
}

CommandLineArgs parseCommandLine(int argc, char* argv[], int rank) {
    CommandLineArgs args;

    args.test_case = 0;
    args.t_final = T_FINAL;
    args.cfl = CFL;
    args.s_l = S_L;
    args.u_vel = U_CONST;
    args.v_vel = V_CONST;
    args.w_vel = W_CONST;
    args.enable_reinit = ENABLE_REINIT;
    args.reinit_interval = REINIT_INTERVAL;
    args.reinit_iterations = REINIT_ITERATIONS;
    strcpy(args.output_dir, "./output");
    args.save_output = true;
    args.verbose = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0], rank);
            MPI_Finalize();
            exit(0);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "zalesak") == 0) {
                args.test_case = 1;
            } else {
                args.test_case = 0;  // sphere
            }
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
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            args.w_vel = atof(argv[++i]);
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
            if (rank == 0) {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
            }
            printUsage(argv[0], rank);
            MPI_Finalize();
            exit(1);
        }
    }

    return args;
}

//=============================================================================
// Standalone Simulation (3D)
//=============================================================================

void runStandaloneSimulation3D(const CommandLineArgs& args, int rank, int num_procs) {
    FILE* log_fp = nullptr;

    if (rank == 0) {
        printf("========================================\n");
        printf("G-Equation Level-Set Solver 3D (MPI)\n");
        printf("========================================\n\n");

        log_fp = fopen("log/simulation.log", "w");
        if (log_fp) {
            fprintf(log_fp, "========================================\n");
            fprintf(log_fp, "G-Equation Level-Set Solver 3D (MPI)\n");
            fprintf(log_fp, "========================================\n\n");
        }
    }

    if (rank == 0) {
        printf("MPI Configuration:\n");
        printf("  Number of processes: %d\n", num_procs);
        printf("  Domain decomposition: 1D in Y direction\n\n");
        if (log_fp) {
            fprintf(log_fp, "MPI Configuration:\n");
            fprintf(log_fp, "  Number of processes: %d\n", num_procs);
            fprintf(log_fp, "  Domain decomposition: 1D in Y direction\n\n");
        }
    }

    // Initialize parameters
    SimParams params = initDefaultParams();
    setupMPIDomainDecomposition(params, rank, num_procs);

    // Override with command line arguments
    params.s_l = args.s_l;
    params.u_const = args.u_vel;
    params.v_const = args.v_vel;
    params.w_const = args.w_vel;
    params.t_final = args.t_final;
    params.cfl = args.cfl;
    params.enable_reinit = args.enable_reinit;
    params.reinit_interval = args.reinit_interval;
    params.reinit_iterations = args.reinit_iterations;

    // Calculate time step
    params.dt = calculateTimeStep(params);

    printParameters3D(params);

    if (rank == 0 && log_fp) {
        fprintf(log_fp, "Simulation Parameters:\n");
        fprintf(log_fp, "  Grid size: %d x %d x %d\n", params.nx, params.ny, params.nz);
        fprintf(log_fp, "  dx: %.6e, dy: %.6e, dz: %.6e\n", params.dx, params.dy, params.dz);
        fprintf(log_fp, "  dt: %.6e\n", params.dt);
        fprintf(log_fp, "  Final time: %.4f\n", params.t_final);
        fprintf(log_fp, "  CFL: %.2f\n", params.cfl);
        fprintf(log_fp, "  Velocity: u=%.2f, v=%.2f, w=%.2f\n",
                params.u_const, params.v_const, params.w_const);
        fprintf(log_fp, "  Flame speed (S_L): %.2f\n", params.s_l);
        fprintf(log_fp, "  Reinitialization: %s\n", params.enable_reinit ? "Enabled" : "Disabled");
        fprintf(log_fp, "\n");
    }

    // Allocate memory
    long long local_total_size = (long long)params.nx_total * params.local_ny_total * params.nz_total;

    if (rank == 0) {
        double mem_mb = local_total_size * sizeof(double) * 12.0 / (1024.0 * 1024.0);
        printf("Allocating memory: %d x %d x %d = %lld cells per process (%.2f MB)\n",
               params.nx_total, params.local_ny_total, params.nz_total,
               local_total_size, mem_mb);
    }

    double* G             = new double[local_total_size]();
    double* G_new         = new double[local_total_size]();
    double* G_1           = new double[local_total_size]();
    double* G_2           = new double[local_total_size]();
    double* G_rhs         = new double[local_total_size]();
    double* G_initial     = new double[local_total_size]();
    double* u             = new double[local_total_size]();
    double* v             = new double[local_total_size]();
    double* w             = new double[local_total_size]();
    double* G_reinit_temp = new double[local_total_size]();
    double* G_reinit_0    = new double[local_total_size]();
    double* r_tilde       = new double[local_total_size]();
    int*    interface_flag = new int[local_total_size]();

    // Initialize test case
    if (rank == 0) {
        printf("Initializing test case...\n");
    }

    if (args.test_case == 0) {
        initSphereTestCase(G, u, v, w, params);
        if (rank == 0) {
            printf("Test case: Sphere Deformation\n");
            if (log_fp) fprintf(log_fp, "Test case: Sphere Deformation\n");
        }
    } else if (args.test_case == 1) {
        initZalesak3DTestCase(G, u, v, w, params);
        if (rank == 0) {
            printf("Test case: Zalesak's Slotted Sphere Rotation\n");
            if (log_fp) fprintf(log_fp, "Test case: Zalesak's Slotted Sphere Rotation\n");
        }
    }

    // Apply boundary conditions
    applyBoundaryConditions3D(G, params);
    applyVelocityBoundaryConditions3D(u, v, w, params);

    // Store initial field
    copyInitialField(G, G_initial, local_total_size);

    // Calculate initial volume
    double initial_volume = computeInterfaceVolume(G, params);
    if (rank == 0) {
        printf("Initial interface volume: %.6f\n\n", initial_volume);
        if (log_fp) fprintf(log_fp, "Initial interface volume: %.6f\n\n", initial_volume);
    }

    // Save initial field (time = 0)
    if (args.save_output) {
        saveFieldBinary3D("output/G_initial.bin", G, params, 0.0);
        if (rank == 0) {
            printf("Saved initial field to output/G_initial.bin\n");
        }
    }

    // Main time loop
    if (rank == 0) {
        printf("\nStarting simulation...\n");
        printf("Target time: %.4f, dt = %.6e\n\n", params.t_final, params.dt);
    }

    double start_time = MPI_Wtime();

    double current_time = 0.0;
    int step = 0;
    int max_steps = (int)(params.t_final / params.dt) + 100;

    while (current_time < params.t_final && step < max_steps) {
        if (current_time + params.dt > params.t_final) {
            params.dt = params.t_final - current_time;
        }

        // Update velocity field for current time (deformation test)
        params.current_time = current_time;
        updateDeformationVelocity(u, v, w, params);
        applyVelocityBoundaryConditions3D(u, v, w, params);

        // RK3 time step
        rk3TimeStep3D(G, G_new, G_1, G_2, G_rhs, u, v, w, params);

        // Swap
        double* temp = G;
        G = G_new;
        G_new = temp;

        // Reinitialization
        if (params.enable_reinit && step > 0 && step % params.reinit_interval == 0) {
            reinitializeWithSwap3D(&G, &G_reinit_temp, G_reinit_0,
                                    r_tilde, interface_flag, params);
        }

        current_time += params.dt;
        step++;

        // Progress
        if (step % OUTPUT_INTERVAL == 0) {
            double l2_error = computeL2Error3D(G, G_initial, params);
            double volume = computeInterfaceVolume(G, params);

            if (args.verbose) {
                printProgress3D(step, current_time, params.dt, l2_error, volume, rank);
            }

            if (args.save_output) {
                char snapshot_path[300];
                snprintf(snapshot_path, sizeof(snapshot_path),
                         "%s/G_step_%06d.bin", args.output_dir, step);
                saveFieldBinary3D(snapshot_path, G, params, current_time);
            }
        }
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Final results
    double final_l2_error = computeL2Error3D(G, G_initial, params);
    double final_volume = computeInterfaceVolume(G, params);
    double shape_error = computeMeanShapeError3D(G, G_initial, params);

    printSummary3D(final_l2_error, shape_error, initial_volume, final_volume, rank);

    if (rank == 0) {
        printf("\n========================================\n");
        printf("Timing Results:\n");
        printf("  Total steps: %d\n", step);
        printf("  Wall time: %.3f s\n", max_elapsed);
        printf("  Time per step: %.3f ms\n", max_elapsed / step * 1000.0);
        printf("  Processes: %d\n", num_procs);
        printf("========================================\n");

        if (log_fp) {
            fprintf(log_fp, "\nFinal Results:\n");
            fprintf(log_fp, "  L2 Error: %.6e\n", final_l2_error);
            fprintf(log_fp, "  Shape Error: %.6e\n", shape_error);
            fprintf(log_fp, "  Initial Volume: %.6f\n", initial_volume);
            fprintf(log_fp, "  Final Volume: %.6f\n", final_volume);
            fprintf(log_fp, "  Volume Change: %.6f%%\n",
                    100.0 * (final_volume - initial_volume) / initial_volume);
            fprintf(log_fp, "\nTiming Results:\n");
            fprintf(log_fp, "  Total steps: %d\n", step);
            fprintf(log_fp, "  Wall time: %.3f s\n", max_elapsed);
            fprintf(log_fp, "  Time per step: %.3f ms\n", max_elapsed / step * 1000.0);
            fprintf(log_fp, "  Processes: %d\n", num_procs);
            fprintf(log_fp, "========================================\n");
            fclose(log_fp);
        }
    }

    // Save final field
    if (args.save_output) {
        saveFieldBinary3D("output/G_final.bin", G, params, current_time);
        saveFieldVTK3D("output/G_final.vtk", G, params);
        if (rank == 0) {
            printf("\nSaved final field to output/G_final.bin and output/G_final.vtk\n");
        }
    }

    // Cleanup
    delete[] G;
    delete[] G_new;
    delete[] G_1;
    delete[] G_2;
    delete[] G_rhs;
    delete[] G_initial;
    delete[] u;
    delete[] v;
    delete[] w;
    delete[] G_reinit_temp;
    delete[] G_reinit_0;
    delete[] r_tilde;
    delete[] interface_flag;

    if (rank == 0) {
        printf("\nSimulation complete.\n");
    }
}

//=============================================================================
// Main Entry Point
//=============================================================================

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    CommandLineArgs args = parseCommandLine(argc, argv, rank);

    if (args.save_output && rank == 0) {
        char cmd[300];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", args.output_dir);
        int result = system(cmd);
        (void)result;
        int log_result = system("mkdir -p log");
        (void)log_result;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    runStandaloneSimulation3D(args, rank, num_procs);

    MPI_Finalize();

    return 0;
}
