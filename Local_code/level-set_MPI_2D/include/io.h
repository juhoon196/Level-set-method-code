/**
 * @file io.h
 * @brief Input/Output and error calculation utilities (MPI version)
 *
 * This module provides:
 * - Binary file I/O for G field (with MPI gather)
 * - VTK output for visualization
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
// Binary File I/O
//=============================================================================

/**
 * @brief Save G field to binary file (MPI version - rank 0 gathers and writes)
 *
 * @param filename Output filename
 * @param local_G Local G field array
 * @param params Simulation parameters
 * @return true on success, false on failure
 */
inline bool saveFieldBinary(const char* filename, const double* local_G, SimParams& params) {
    int nx_total = params.nx_total;
    // int local_ny_total = params.local_ny_total; // unused
    int nghost = params.nghost;
    int local_ny = params.local_ny;

    // Each process sends only its interior rows (not ghost cells)
    int local_interior_size = nx_total * local_ny;
    std::vector<double> local_interior(local_interior_size);

    // Copy interior rows to send buffer
    for (int j = 0; j < local_ny; j++) {
        for (int i = 0; i < nx_total; i++) {
            local_interior[j * nx_total + i] = local_G[(j + nghost) * nx_total + i];
        }
    }

    // Gather sizes from all processes
    std::vector<int> recv_counts(params.num_procs);
    std::vector<int> displs(params.num_procs);

    MPI_Gather(&local_interior_size, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (params.rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < params.num_procs; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }

    // Total size for global array
    int global_size = params.nx_total * params.ny;
    std::vector<double> global_G;
    if (params.rank == 0) {
        global_G.resize(global_size);
    }

    // Gather all data to rank 0
    MPI_Gatherv(local_interior.data(), local_interior_size, MPI_DOUBLE,
                global_G.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0 writes the file
    if (params.rank == 0) {
        FILE* fp = fopen(filename, "wb");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
            return false;
        }

        // Write header
        int header[3] = {params.nx, params.ny, params.nghost};
        fwrite(header, sizeof(int), 3, fp);

        // Need to reconstruct full array with ghost cells for compatibility
        int ny_total = params.ny + 2 * nghost;
        std::vector<double> full_G(nx_total * ny_total, 0.0);

        // Copy interior data
        for (int j = 0; j < params.ny; j++) {
            for (int i = 0; i < nx_total; i++) {
                full_G[(j + nghost) * nx_total + i] = global_G[j * nx_total + i];
            }
        }

        // Fill ghost cells (simple copy from adjacent)
        for (int i = 0; i < nx_total; i++) {
            for (int g = 0; g < nghost; g++) {
                full_G[g * nx_total + i] = full_G[nghost * nx_total + i];
                full_G[(params.ny + nghost + g) * nx_total + i] = 
                    full_G[(params.ny + nghost - 1) * nx_total + i];
            }
        }

        fwrite(full_G.data(), sizeof(double), nx_total * ny_total, fp);
        fclose(fp);
    }

    return true;
}

/**
 * @brief Save G field in VTK format for visualization
 */
inline bool saveFieldVTK(const char* filename, const double* local_G, SimParams& params) {
    int nx_total = params.nx_total;
    int local_ny = params.local_ny;
    int nghost = params.nghost;

    // Gather interior data to rank 0
    int local_interior_size = nx_total * local_ny;
    std::vector<double> local_interior(local_interior_size);

    for (int j = 0; j < local_ny; j++) {
        for (int i = 0; i < nx_total; i++) {
            local_interior[j * nx_total + i] = local_G[(j + nghost) * nx_total + i];
        }
    }

    std::vector<int> recv_counts(params.num_procs);
    std::vector<int> displs(params.num_procs);

    MPI_Gather(&local_interior_size, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (params.rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < params.num_procs; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }

    int global_size = nx_total * params.ny;
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
        fprintf(fp, "G-equation Level-Set Field\n");
        fprintf(fp, "ASCII\n");
        fprintf(fp, "DATASET STRUCTURED_POINTS\n");
        fprintf(fp, "DIMENSIONS %d %d 1\n", params.nx, params.ny);
        fprintf(fp, "ORIGIN %f %f 0\n", params.x_min, params.y_min);
        fprintf(fp, "SPACING %f %f 1\n", params.dx, params.dy);
        fprintf(fp, "POINT_DATA %d\n", params.nx * params.ny);
        fprintf(fp, "SCALARS G double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");

        for (int j = 0; j < params.ny; j++) {
            for (int i = nghost; i < params.nx + nghost; i++) {
                fprintf(fp, "%e\n", global_G[j * nx_total + i]);
            }
        }

        fclose(fp);
    }

    return true;
}

//=============================================================================
// Error Calculation (MPI version)
//=============================================================================

/**
 * @brief Compute L2 error between current and reference G field (MPI version)
 */
inline double computeL2Error(const double* G, const double* G_ref, SimParams& params) {
    int nghost = params.nghost;
    int nx = params.nx;
    int local_ny = params.local_ny;
    int nx_total = params.nx_total;

    double local_sum = 0.0;
    int local_count = 0;

    for (int j = nghost; j < local_ny + nghost; j++) {
        for (int i = nghost; i < nx + nghost; i++) {
            int index = idx(i, j, nx_total);
            double diff = G[index] - G_ref[index];
            local_sum += diff * diff;
            local_count++;
        }
    }

    // MPI Allreduce to sum across all processes
    double global_sum = 0.0;
    int global_count = 0;

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return sqrt(global_sum / global_count);
}

/**
 * @brief Compute interface area (region where G < 0) (MPI version)
 */
inline double computeInterfaceArea(const double* G, SimParams& params) {
    int nghost = params.nghost;
    int nx = params.nx;
    int local_ny = params.local_ny;
    int nx_total = params.nx_total;

    double local_area = 0.0;

    for (int j = nghost; j < local_ny + nghost; j++) {
        for (int i = nghost; i < nx + nghost; i++) {
            int index = idx(i, j, nx_total);
            if (G[index] < 0.0) {
                local_area += params.dx * params.dy;
            }
        }
    }

    double global_area = 0.0;
    MPI_Allreduce(&local_area, &global_area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_area;
}

/**
 * @brief Compute mean shape error (relative area difference)
 */
inline double computeMeanShapeError(const double* G, const double* G_initial, SimParams& params) {
    double area_current = computeInterfaceArea(G, params);
    double area_initial = computeInterfaceArea(G_initial, params);

    if (area_initial < 1e-15) return 0.0;

    return fabs(area_current - area_initial) / area_initial;
}

//=============================================================================
// Console Output Utilities
//=============================================================================

/**
 * @brief Print simulation parameters (only rank 0)
 */
inline void printParameters(const SimParams& params) {
    if (params.rank != 0) return;

    printf("========================================\n");
    printf("G-Equation Level-Set Solver (MPI Version)\n");
    printf("========================================\n");
    printf("MPI:\n");
    printf("  Number of processes: %d\n", params.num_procs);
    printf("\nGrid:\n");
    printf("  Global interior points: %d x %d\n", params.nx, params.ny);
    printf("  Local rows (rank 0): %d\n", params.local_ny);
    printf("  Ghost cells: %d\n", params.nghost);
    printf("  Domain: [%.2f, %.2f] x [%.2f, %.2f]\n",
           params.x_min, params.x_max, params.y_min, params.y_max);
    printf("  Grid spacing: dx=%.6f, dy=%.6f\n", params.dx, params.dy);
    printf("\nPhysics:\n");
    printf("  Laminar flame speed: S_L = %.4f\n", params.s_l);
    printf("  Advection velocity: (u, v) = (%.4f, %.4f)\n",
           params.u_const, params.v_const);
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

/**
 * @brief Print progress during simulation (only rank 0)
 */
inline void printProgress(int step, double t, double dt, double l2_error, double area, int rank) {
    if (rank != 0) return;
    printf("Step %6d: t = %.6f, dt = %.6e, L2 error = %.6e, Area = %.6f\n",
           step, t, dt, l2_error, area);
}

/**
 * @brief Print final summary (only rank 0)
 */
inline void printSummary(double l2_error, double shape_error, double area_initial, double area_final, int rank) {
    if (rank != 0) return;

    printf("\n========================================\n");
    printf("Simulation Complete\n");
    printf("========================================\n");
    printf("Final L2 Error: %.6e\n", l2_error);
    printf("Mean Shape Error: %.4f%%\n", shape_error * 100.0);
    printf("Initial Area: %.6f\n", area_initial);
    printf("Final Area: %.6f\n", area_final);
    printf("Area Conservation Error: %.4f%%\n",
           fabs(area_final - area_initial) / area_initial * 100.0);
    printf("========================================\n");
}

#endif // IO_H
