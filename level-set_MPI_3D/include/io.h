/**
 * @file io.h
 * @brief Input/Output and error calculation utilities (MPI 3D version)
 *
 * Provides:
 * - Binary file I/O with MPI gather
 * - VTK structured points output for 3D visualization
 * - L2 error calculation with MPI reduction
 * - Console output utilities
 */

#ifndef IO_H
#define IO_H

#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "config.h"

//=============================================================================
// Binary File I/O (3D)
//=============================================================================

/**
 * @brief Save G field to binary file (MPI 3D version - rank 0 gathers and writes)
 *
 * Binary format:
 *   int[4]   : nx, ny, nz, nghost
 *   double[4]: time, dx, dy, dz
 *   double[nx_total * ny_total * nz_total] : field data (with ghost cells)
 */
inline bool saveFieldBinary3D(const char* filename, const double* local_G,
                               SimParams& params, double current_time = 0.0) {
    int nx_total = params.nx_total;
    int nz_total = params.nz_total;
    int nghost = params.nghost;
    int local_ny = params.local_ny;
    int local_ny_total = params.local_ny_total;

    // Each process sends interior data (excluding y ghost cells)
    int local_interior_size = nx_total * local_ny * nz_total;
    std::vector<double> local_interior(local_interior_size);

    // Copy interior rows
    int pos = 0;
    for (int k = 0; k < nz_total; k++) {
        for (int j = 0; j < local_ny; j++) {
            for (int i = 0; i < nx_total; i++) {
                local_interior[pos++] = local_G[idx3(i, j + nghost, k, nx_total, local_ny_total)];
            }
        }
    }

    // Gather sizes
    std::vector<int> recv_counts(params.num_procs);
    std::vector<int> displs(params.num_procs);

    MPI_Gather(&local_interior_size, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (params.rank == 0) {
        displs[0] = 0;
        for (int p = 1; p < params.num_procs; p++) {
            displs[p] = displs[p - 1] + recv_counts[p - 1];
        }
    }

    int global_size = nx_total * params.ny * nz_total;
    std::vector<double> global_G;
    if (params.rank == 0) {
        global_G.resize(global_size);
    }

    MPI_Gatherv(local_interior.data(), local_interior_size, MPI_DOUBLE,
                global_G.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (params.rank == 0) {
        FILE* fp = fopen(filename, "wb");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
            return false;
        }

        // Write header: ints then doubles
        int header_i[4] = {params.nx, params.ny, params.nz, params.nghost};
        fwrite(header_i, sizeof(int), 4, fp);
        double header_d[4] = {current_time, params.dx, params.dy, params.dz};
        fwrite(header_d, sizeof(double), 4, fp);

        // Reconstruct full array with ghost cells in y
        int ny_total_g = params.ny + 2 * nghost;
        std::vector<double> full_G(nx_total * ny_total_g * nz_total, 0.0);

        // Copy interior
        for (int k = 0; k < nz_total; k++) {
            for (int j = 0; j < params.ny; j++) {
                for (int i = 0; i < nx_total; i++) {
                    int src = k * nx_total * params.ny + j * nx_total + i;
                    int dst = k * nx_total * ny_total_g + (j + nghost) * nx_total + i;
                    full_G[dst] = global_G[src];
                }
            }
        }

        // Fill y ghost cells
        for (int k = 0; k < nz_total; k++) {
            for (int i = 0; i < nx_total; i++) {
                for (int g = 0; g < nghost; g++) {
                    int bot_src = k * nx_total * ny_total_g + nghost * nx_total + i;
                    int bot_dst = k * nx_total * ny_total_g + g * nx_total + i;
                    full_G[bot_dst] = full_G[bot_src];

                    int top_src = k * nx_total * ny_total_g + (params.ny + nghost - 1) * nx_total + i;
                    int top_dst = k * nx_total * ny_total_g + (params.ny + nghost + g) * nx_total + i;
                    full_G[top_dst] = full_G[top_src];
                }
            }
        }

        fwrite(full_G.data(), sizeof(double), nx_total * ny_total_g * nz_total, fp);
        fclose(fp);
    }

    return true;
}

//=============================================================================
// VTK Output (3D)
//=============================================================================

/**
 * @brief Save G field in VTK format for 3D visualization
 */
inline bool saveFieldVTK3D(const char* filename, const double* local_G, SimParams& params) {
    int nx_total = params.nx_total;
    int nz_total = params.nz_total;
    int nghost = params.nghost;
    int local_ny = params.local_ny;
    int local_ny_total = params.local_ny_total;

    // Gather interior data
    int local_interior_size = nx_total * local_ny * nz_total;
    std::vector<double> local_interior(local_interior_size);

    int pos = 0;
    for (int k = 0; k < nz_total; k++) {
        for (int j = 0; j < local_ny; j++) {
            for (int i = 0; i < nx_total; i++) {
                local_interior[pos++] = local_G[idx3(i, j + nghost, k, nx_total, local_ny_total)];
            }
        }
    }

    std::vector<int> recv_counts(params.num_procs);
    std::vector<int> displs(params.num_procs);

    MPI_Gather(&local_interior_size, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (params.rank == 0) {
        displs[0] = 0;
        for (int p = 1; p < params.num_procs; p++) {
            displs[p] = displs[p - 1] + recv_counts[p - 1];
        }
    }

    int global_size = nx_total * params.ny * nz_total;
    std::vector<double> global_G;
    if (params.rank == 0) {
        global_G.resize(global_size);
    }

    MPI_Gatherv(local_interior.data(), local_interior_size, MPI_DOUBLE,
                global_G.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (params.rank == 0) {
        FILE* fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
            return false;
        }

        fprintf(fp, "# vtk DataFile Version 3.0\n");
        fprintf(fp, "G-equation Level-Set Field 3D\n");
        fprintf(fp, "ASCII\n");
        fprintf(fp, "DATASET STRUCTURED_POINTS\n");
        fprintf(fp, "DIMENSIONS %d %d %d\n", params.nx, params.ny, params.nz);
        fprintf(fp, "ORIGIN %f %f %f\n", params.x_min, params.y_min, params.z_min);
        fprintf(fp, "SPACING %f %f %f\n", params.dx, params.dy, params.dz);
        fprintf(fp, "POINT_DATA %d\n", params.nx * params.ny * params.nz);
        fprintf(fp, "SCALARS G double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");

        // VTK structured points: x varies fastest, then y, then z
        for (int k = nghost; k < params.nz + nghost; k++) {
            for (int j = 0; j < params.ny; j++) {
                for (int i = nghost; i < params.nx + nghost; i++) {
                    int gidx = k * nx_total * params.ny + j * nx_total + i;
                    fprintf(fp, "%e\n", global_G[gidx]);
                }
            }
        }

        fclose(fp);
    }

    return true;
}

//=============================================================================
// Error Calculation (MPI 3D version)
//=============================================================================

/**
 * @brief Compute L2 error between current and reference G field (3D)
 */
inline double computeL2Error3D(const double* G, const double* G_ref, SimParams& params) {
    int nghost = params.nghost;
    int nx = params.nx;
    int nz = params.nz;
    int local_ny = params.local_ny;
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    double local_sum = 0.0;
    int local_count = 0;

    for (int k = nghost; k < nz + nghost; k++) {
        for (int j = nghost; j < local_ny + nghost; j++) {
            for (int i = nghost; i < nx + nghost; i++) {
                int index = idx3(i, j, k, nx_total, local_ny_total);
                double diff = G[index] - G_ref[index];
                local_sum += diff * diff;
                local_count++;
            }
        }
    }

    double global_sum = 0.0;
    int global_count = 0;

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return sqrt(global_sum / global_count);
}

/**
 * @brief Compute interface volume (region where G < 0) (3D)
 */
inline double computeInterfaceVolume(const double* G, SimParams& params) {
    int nghost = params.nghost;
    int nx = params.nx;
    int nz = params.nz;
    int local_ny = params.local_ny;
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    double local_vol = 0.0;

    for (int k = nghost; k < nz + nghost; k++) {
        for (int j = nghost; j < local_ny + nghost; j++) {
            for (int i = nghost; i < nx + nghost; i++) {
                int index = idx3(i, j, k, nx_total, local_ny_total);
                if (G[index] < 0.0) {
                    local_vol += params.dx * params.dy * params.dz;
                }
            }
        }
    }

    double global_vol = 0.0;
    MPI_Allreduce(&local_vol, &global_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_vol;
}

/**
 * @brief Compute mean shape error (relative volume difference)
 */
inline double computeMeanShapeError3D(const double* G, const double* G_initial, SimParams& params) {
    double vol_current = computeInterfaceVolume(G, params);
    double vol_initial = computeInterfaceVolume(G_initial, params);

    if (vol_initial < 1e-15) return 0.0;

    return fabs(vol_current - vol_initial) / vol_initial;
}

//=============================================================================
// Console Output Utilities (3D)
//=============================================================================

inline void printParameters3D(const SimParams& params) {
    if (params.rank != 0) return;

    printf("========================================\n");
    printf("G-Equation Level-Set Solver 3D (MPI)\n");
    printf("========================================\n");
    printf("MPI:\n");
    printf("  Number of processes: %d\n", params.num_procs);
    printf("\nGrid:\n");
    printf("  Global interior points: %d x %d x %d\n", params.nx, params.ny, params.nz);
    printf("  Local rows (rank 0): %d\n", params.local_ny);
    printf("  Ghost cells: %d\n", params.nghost);
    printf("  Domain: [%.2f, %.2f] x [%.2f, %.2f] x [%.2f, %.2f]\n",
           params.x_min, params.x_max, params.y_min, params.y_max,
           params.z_min, params.z_max);
    printf("  Grid spacing: dx=%.6f, dy=%.6f, dz=%.6f\n", params.dx, params.dy, params.dz);
    printf("\nPhysics:\n");
    printf("  Laminar flame speed: S_L = %.4f\n", params.s_l);
    printf("  Advection velocity: (u, v, w) = (%.4f, %.4f, %.4f)\n",
           params.u_const, params.v_const, params.w_const);
    printf("\nTime integration:\n");
    printf("  CFL number: %.4f\n", params.cfl);
    printf("  Time step: dt = %.6e\n", params.dt);
    printf("  Final time: T = %.4f\n", params.t_final);
    printf("\nReinitialization:\n");
    printf("  Enabled: %s\n", params.enable_reinit ? "Yes" : "No");
    if (params.enable_reinit) {
        printf("  Interval: every %d steps\n", params.reinit_interval);
        printf("  Iterations: %d\n", params.reinit_iterations);
        printf("  Pseudo-time step: %.6e\n", params.reinit_dtau);
        printf("  Beta (forcing): %.4f\n", params.reinit_beta);
    }
    printf("========================================\n\n");
}

inline void printProgress3D(int step, double t, double dt, double l2_error, double volume, int rank) {
    if (rank != 0) return;
    printf("Step %6d: t = %.6f, dt = %.6e, L2 error = %.6e, Volume = %.6f\n",
           step, t, dt, l2_error, volume);
}

inline void printSummary3D(double l2_error, double shape_error,
                            double vol_initial, double vol_final, int rank) {
    if (rank != 0) return;

    printf("\n========================================\n");
    printf("Simulation Complete\n");
    printf("========================================\n");
    printf("Final L2 Error: %.6e\n", l2_error);
    printf("Mean Shape Error: %.4f%%\n", shape_error * 100.0);
    printf("Initial Volume: %.6f\n", vol_initial);
    printf("Final Volume: %.6f\n", vol_final);
    printf("Volume Conservation Error: %.4f%%\n",
           fabs(vol_final - vol_initial) / vol_initial * 100.0);
    printf("========================================\n");
}

#endif // IO_H
