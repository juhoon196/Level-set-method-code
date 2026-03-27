/**
 * @file io.cuh
 * @brief Input/Output and error calculation utilities
 *
 * This module provides:
 * - Binary file I/O for G field
 * - VTK output for visualization
 * - L2 error calculation for convergence tests
 * - Console output utilities
 */

#ifndef IO_CUH
#define IO_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "config.cuh"

//=============================================================================
// Binary File I/O
//=============================================================================

/**
 * @brief Save G field to binary file
 *
 * File format:
 * - Header: nx, ny, nghost (3 integers)
 * - Data: G values in row-major order (including ghost cells)
 *
 * @param filename Output filename
 * @param h_G Host array containing G field
 * @param params Simulation parameters
 * @return true on success, false on failure
 */
bool saveFieldBinary(const char* filename, const double* h_G, const SimParams& params) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return false;
    }

    // Write header
    int header[3] = {params.nx, params.ny, params.nghost};
    size_t written = fwrite(header, sizeof(int), 3, fp);
    if (written != 3) {
        fprintf(stderr, "Error: Failed to write header\n");
        fclose(fp);
        return false;
    }

    // Write data
    int total_size = params.nx_total * params.ny_total;
    written = fwrite(h_G, sizeof(double), total_size, fp);
    if (written != (size_t)total_size) {
        fprintf(stderr, "Error: Failed to write data\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

/**
 * @brief Save interior G field only (without ghost cells)
 *
 * @param filename Output filename
 * @param h_G Host array containing G field (with ghost cells)
 * @param params Simulation parameters
 * @return true on success, false on failure
 */
bool saveInteriorFieldBinary(const char* filename, const double* h_G, const SimParams& params) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return false;
    }

    // Write header
    int header[2] = {params.nx, params.ny};
    size_t written = fwrite(header, sizeof(int), 2, fp);
    if (written != 2) {
        fprintf(stderr, "Error: Failed to write header\n");
        fclose(fp);
        return false;
    }

    // Write interior data only
    for (int j = params.nghost; j < params.ny + params.nghost; j++) {
        for (int i = params.nghost; i < params.nx + params.nghost; i++) {
            double val = h_G[idx(i, j, params.nx_total)];
            fwrite(&val, sizeof(double), 1, fp);
        }
    }

    fclose(fp);
    return true;
}

/**
 * @brief Load G field from binary file
 *
 * @param filename Input filename
 * @param h_G Host array to store G field
 * @param params Simulation parameters (used to verify dimensions)
 * @return true on success, false on failure
 */
bool loadFieldBinary(const char* filename, double* h_G, const SimParams& params) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        return false;
    }

    // Read header
    int header[3];
    size_t read = fread(header, sizeof(int), 3, fp);
    if (read != 3) {
        fprintf(stderr, "Error: Failed to read header\n");
        fclose(fp);
        return false;
    }

    // Verify dimensions
    if (header[0] != params.nx || header[1] != params.ny || header[2] != params.nghost) {
        fprintf(stderr, "Error: Dimension mismatch in file %s\n", filename);
        fprintf(stderr, "  Expected: nx=%d, ny=%d, nghost=%d\n",
                params.nx, params.ny, params.nghost);
        fprintf(stderr, "  Got: nx=%d, ny=%d, nghost=%d\n",
                header[0], header[1], header[2]);
        fclose(fp);
        return false;
    }

    // Read data
    int total_size = params.nx_total * params.ny_total;
    read = fread(h_G, sizeof(double), total_size, fp);
    if (read != (size_t)total_size) {
        fprintf(stderr, "Error: Failed to read data\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

//=============================================================================
// VTK Output for Visualization
//=============================================================================

/**
 * @brief Save G field in VTK format for visualization (Paraview, VisIt, etc.)
 *
 * @param filename Output filename (should end with .vtk)
 * @param h_G Host array containing G field
 * @param params Simulation parameters
 * @return true on success, false on failure
 */
bool saveFieldVTK(const char* filename, const double* h_G, const SimParams& params) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return false;
    }

    // VTK header
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

    // Write interior data only
    for (int j = params.nghost; j < params.ny + params.nghost; j++) {
        for (int i = params.nghost; i < params.nx + params.nghost; i++) {
            fprintf(fp, "%e\n", h_G[idx(i, j, params.nx_total)]);
        }
    }

    fclose(fp);
    return true;
}

//=============================================================================
// Error Calculation
//=============================================================================

/**
 * @brief CUDA kernel to compute squared error
 */
__global__ void computeSquaredErrorKernel(const double* __restrict__ G,
                                           const double* __restrict__ G_ref,
                                           double* __restrict__ error_sq,
                                           SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only compute for interior points
    if (i < params.nghost || i >= params.nx + params.nghost ||
        j < params.nghost || j >= params.ny + params.nghost) {
        return;
    }

    int index = idx(i, j, params.nx_total);
    double diff = G[index] - G_ref[index];
    error_sq[index] = diff * diff;
}

/**
 * @brief CUDA kernel for parallel reduction (sum)
 */
__global__ void reduceSum(const double* __restrict__ input,
                          double* __restrict__ output,
                          int n) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and do first level of reduction
    double sum = 0.0;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Compute L2 error between current and reference G field
 *
 * L2 error = sqrt(sum((G - G_ref)^2) / N)
 *
 * @param d_G Current G field (device)
 * @param d_G_ref Reference G field (device)
 * @param d_temp Temporary array for reduction
 * @param params Simulation parameters
 * @return L2 error (normalized by number of points)
 */
double computeL2Error(const double* d_G, const double* d_G_ref,
                      double* d_temp, SimParams& params) {
    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    int total_size = params.nx_total * params.ny_total;

    // Compute squared errors
    computeSquaredErrorKernel<<<grid, block>>>(d_G, d_G_ref, d_temp, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy to host and sum (simple approach for correctness)
    double* h_temp = new double[total_size];
    CUDA_CHECK(cudaMemcpy(h_temp, d_temp, total_size * sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    int count = 0;
    for (int j = params.nghost; j < params.ny + params.nghost; j++) {
        for (int i = params.nghost; i < params.nx + params.nghost; i++) {
            sum += h_temp[idx(i, j, params.nx_total)];
            count++;
        }
    }

    delete[] h_temp;

    return sqrt(sum / count);
}

/**
 * @brief Compute interface shape error (area difference for closed curves)
 *
 * Shape error is computed as the difference in interface area/length
 * based on the zero level-set contour.
 */
__global__ void computeInterfaceAreaKernel(const double* __restrict__ G,
                                            double* __restrict__ area,
                                            SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < params.nghost || i >= params.nx + params.nghost ||
        j < params.nghost || j >= params.ny + params.nghost) {
        return;
    }

    int index = idx(i, j, params.nx_total);

    // Count cells where G < 0 (inside interface)
    area[index] = (G[index] < 0.0) ? params.dx * params.dy : 0.0;
}

/**
 * @brief Compute interface area (region where G < 0)
 */
double computeInterfaceArea(const double* d_G, double* d_temp, SimParams& params) {
    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    int total_size = params.nx_total * params.ny_total;

    computeInterfaceAreaKernel<<<grid, block>>>(d_G, d_temp, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum on host
    double* h_temp = new double[total_size];
    CUDA_CHECK(cudaMemcpy(h_temp, d_temp, total_size * sizeof(double), cudaMemcpyDeviceToHost));

    double area = 0.0;
    for (int j = params.nghost; j < params.ny + params.nghost; j++) {
        for (int i = params.nghost; i < params.nx + params.nghost; i++) {
            area += h_temp[idx(i, j, params.nx_total)];
        }
    }

    delete[] h_temp;
    return area;
}

/**
 * @brief Compute mean shape error (relative area difference)
 */
double computeMeanShapeError(const double* d_G, const double* d_G_initial,
                             double* d_temp, SimParams& params) {
    double area_current = computeInterfaceArea(d_G, d_temp, params);
    double area_initial = computeInterfaceArea(d_G_initial, d_temp, params);

    if (area_initial < 1e-15) return 0.0;

    return fabs(area_current - area_initial) / area_initial;
}

//=============================================================================
// Console Output Utilities
//=============================================================================

/**
 * @brief Print simulation parameters
 */
void printParameters(const SimParams& params) {
    printf("========================================\n");
    printf("G-Equation Level-Set Solver Parameters\n");
    printf("========================================\n");
    printf("Grid:\n");
    printf("  Interior points: %d x %d\n", params.nx, params.ny);
    printf("  Ghost cells: %d\n", params.nghost);
    printf("  Total size: %d x %d\n", params.nx_total, params.ny_total);
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
 * @brief Print progress during simulation
 */
void printProgress(int step, double t, double dt, double l2_error, double area) {
    printf("Step %6d: t = %.6f, dt = %.6e, L2 error = %.6e, Area = %.6f\n",
           step, t, dt, l2_error, area);
}

/**
 * @brief Print final summary
 */
void printSummary(double l2_error, double shape_error, double area_initial, double area_final) {
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

#endif // IO_CUH
