/**
 * @file boundary.cuh
 * @brief Boundary condition implementations for the G-equation solver
 *
 * This module provides various boundary condition treatments:
 * - Periodic boundary conditions (x-direction as specified)
 * - Zero-gradient (Neumann) boundary conditions
 * - Extrapolation boundary conditions
 *
 * Ghost cells are filled according to the specified boundary conditions
 * to enable WENO-5 stencil computations near boundaries.
 */

#ifndef BOUNDARY_CUH
#define BOUNDARY_CUH

#include <cuda_runtime.h>
#include "config.cuh"

//=============================================================================
// Boundary Condition Types
//=============================================================================

enum BoundaryType {
    BC_PERIODIC,
    BC_ZERO_GRADIENT,
    BC_EXTRAPOLATION,
    BC_DIRICHLET
};

//=============================================================================
// Periodic Boundary Conditions
//=============================================================================

/**
 * @brief Apply periodic boundary conditions in x-direction
 *
 * For WENO-5 with 3 ghost cells on each side:
 * - Left ghost cells copy from right interior cells
 * - Right ghost cells copy from left interior cells
 */
__global__ void applyPeriodicBC_X(double* G, int nx, int ny, int nghost, int nx_total) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= ny + 2 * nghost) return;

    // Left ghost cells (i = 0, 1, 2)
    for (int g = 0; g < nghost; g++) {
        int src_i = nx + g;  // Source from right interior
        int dst_i = g;       // Destination left ghost
        G[idx(dst_i, j, nx_total)] = G[idx(src_i, j, nx_total)];
    }

    // Right ghost cells (i = nx+nghost, nx+nghost+1, nx+nghost+2)
    for (int g = 0; g < nghost; g++) {
        int src_i = nghost + g;         // Source from left interior
        int dst_i = nx + nghost + g;    // Destination right ghost
        G[idx(dst_i, j, nx_total)] = G[idx(src_i, j, nx_total)];
    }
}

/**
 * @brief Apply periodic boundary conditions in y-direction
 */
__global__ void applyPeriodicBC_Y(double* G, int nx, int ny, int nghost, int nx_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx + 2 * nghost) return;

    // Bottom ghost cells
    for (int g = 0; g < nghost; g++) {
        int src_j = ny + g;  // Source from top interior
        int dst_j = g;       // Destination bottom ghost
        G[idx(i, dst_j, nx_total)] = G[idx(i, src_j, nx_total)];
    }

    // Top ghost cells
    for (int g = 0; g < nghost; g++) {
        int src_j = nghost + g;         // Source from bottom interior
        int dst_j = ny + nghost + g;    // Destination top ghost
        G[idx(i, dst_j, nx_total)] = G[idx(i, src_j, nx_total)];
    }
}

//=============================================================================
// Zero-Gradient (Neumann) Boundary Conditions
//=============================================================================

/**
 * @brief Apply zero-gradient BC in y-direction (top and bottom)
 *
 * Ghost cells are set equal to the nearest interior cell
 */
__global__ void applyZeroGradientBC_Y(double* G, int nx, int ny, int nghost, int nx_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx + 2 * nghost) return;

    // Get boundary values
    double bottom_val = G[idx(i, nghost, nx_total)];           // First interior cell
    double top_val = G[idx(i, ny + nghost - 1, nx_total)];     // Last interior cell

    // Fill bottom ghost cells
    for (int g = 0; g < nghost; g++) {
        G[idx(i, g, nx_total)] = bottom_val;
    }

    // Fill top ghost cells
    for (int g = 0; g < nghost; g++) {
        G[idx(i, ny + nghost + g, nx_total)] = top_val;
    }
}

/**
 * @brief Apply zero-gradient BC in x-direction (left and right)
 */
__global__ void applyZeroGradientBC_X(double* G, int nx, int ny, int nghost, int nx_total) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= ny + 2 * nghost) return;

    // Get boundary values
    double left_val = G[idx(nghost, j, nx_total)];             // First interior cell
    double right_val = G[idx(nx + nghost - 1, j, nx_total)];   // Last interior cell

    // Fill left ghost cells
    for (int g = 0; g < nghost; g++) {
        G[idx(g, j, nx_total)] = left_val;
    }

    // Fill right ghost cells
    for (int g = 0; g < nghost; g++) {
        G[idx(nx + nghost + g, j, nx_total)] = right_val;
    }
}

//=============================================================================
// Extrapolation Boundary Conditions
//=============================================================================

/**
 * @brief Apply linear extrapolation BC in y-direction
 *
 * Ghost values are linearly extrapolated from interior cells
 */
__global__ void applyExtrapolationBC_Y(double* G, int nx, int ny, int nghost, int nx_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx + 2 * nghost) return;

    // Bottom extrapolation
    double val0 = G[idx(i, nghost, nx_total)];
    double val1 = G[idx(i, nghost + 1, nx_total)];
    double slope_bottom = val0 - val1;

    for (int g = 0; g < nghost; g++) {
        G[idx(i, nghost - 1 - g, nx_total)] = val0 + (g + 1) * slope_bottom;
    }

    // Top extrapolation
    val0 = G[idx(i, ny + nghost - 1, nx_total)];
    val1 = G[idx(i, ny + nghost - 2, nx_total)];
    double slope_top = val0 - val1;

    for (int g = 0; g < nghost; g++) {
        G[idx(i, ny + nghost + g, nx_total)] = val0 + (g + 1) * slope_top;
    }
}

/**
 * @brief Apply linear extrapolation BC in x-direction
 */
__global__ void applyExtrapolationBC_X(double* G, int nx, int ny, int nghost, int nx_total) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= ny + 2 * nghost) return;

    // Left extrapolation
    double val0 = G[idx(nghost, j, nx_total)];
    double val1 = G[idx(nghost + 1, j, nx_total)];
    double slope_left = val0 - val1;

    for (int g = 0; g < nghost; g++) {
        G[idx(nghost - 1 - g, j, nx_total)] = val0 + (g + 1) * slope_left;
    }

    // Right extrapolation
    val0 = G[idx(nx + nghost - 1, j, nx_total)];
    val1 = G[idx(nx + nghost - 2, j, nx_total)];
    double slope_right = val0 - val1;

    for (int g = 0; g < nghost; g++) {
        G[idx(nx + nghost + g, j, nx_total)] = val0 + (g + 1) * slope_right;
    }
}

//=============================================================================
// Combined Boundary Condition Application
//=============================================================================

/**
 * @brief Apply boundary conditions to G field
 *
 * Default configuration:
 * - X-direction: Periodic (as specified in requirements)
 * - Y-direction: Zero-gradient (Neumann)
 *
 * @param d_G Device pointer to G field
 * @param params Simulation parameters
 */
void applyBoundaryConditions(double* d_G, SimParams& params) {
    // Calculate grid dimensions for boundary kernels
    int threads = 256;
    int blocks_x = (params.ny_total + threads - 1) / threads;
    int blocks_y = (params.nx_total + threads - 1) / threads;

    // Apply periodic BC in x-direction
    applyPeriodicBC_X<<<blocks_x, threads>>>(d_G, params.nx, params.ny,
                                              params.nghost, params.nx_total);
    CUDA_CHECK(cudaGetLastError());

    // Apply zero-gradient BC in y-direction
    applyZeroGradientBC_Y<<<blocks_y, threads>>>(d_G, params.nx, params.ny,
                                                  params.nghost, params.nx_total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Apply fully periodic boundary conditions (both directions)
 */
void applyPeriodicBoundaryConditions(double* d_G, SimParams& params) {
    int threads = 256;
    int blocks_x = (params.ny_total + threads - 1) / threads;
    int blocks_y = (params.nx_total + threads - 1) / threads;

    applyPeriodicBC_X<<<blocks_x, threads>>>(d_G, params.nx, params.ny,
                                              params.nghost, params.nx_total);
    CUDA_CHECK(cudaGetLastError());

    applyPeriodicBC_Y<<<blocks_y, threads>>>(d_G, params.nx, params.ny,
                                              params.nghost, params.nx_total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Apply extrapolation boundary conditions (both directions)
 */
void applyExtrapolationBoundaryConditions(double* d_G, SimParams& params) {
    int threads = 256;
    int blocks_x = (params.ny_total + threads - 1) / threads;
    int blocks_y = (params.nx_total + threads - 1) / threads;

    applyExtrapolationBC_X<<<blocks_x, threads>>>(d_G, params.nx, params.ny,
                                                   params.nghost, params.nx_total);
    CUDA_CHECK(cudaGetLastError());

    applyExtrapolationBC_Y<<<blocks_y, threads>>>(d_G, params.nx, params.ny,
                                                   params.nghost, params.nx_total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

//=============================================================================
// Velocity Field Boundary Conditions
//=============================================================================

/**
 * @brief Apply boundary conditions to velocity field components
 *
 * For constant velocity field, ghost cells are simply filled with the
 * constant values. For variable velocity fields, similar treatment as G.
 *
 * @param d_u X-velocity component (device)
 * @param d_v Y-velocity component (device)
 * @param params Simulation parameters
 */
void applyVelocityBoundaryConditions(double* d_u, double* d_v, SimParams& params) {
    // For constant velocity field, use zero-gradient (copies interior values)
    int threads = 256;
    int blocks_x = (params.ny_total + threads - 1) / threads;
    int blocks_y = (params.nx_total + threads - 1) / threads;

    // Apply to u-velocity
    applyPeriodicBC_X<<<blocks_x, threads>>>(d_u, params.nx, params.ny,
                                              params.nghost, params.nx_total);
    applyZeroGradientBC_Y<<<blocks_y, threads>>>(d_u, params.nx, params.ny,
                                                  params.nghost, params.nx_total);

    // Apply to v-velocity
    applyPeriodicBC_X<<<blocks_x, threads>>>(d_v, params.nx, params.ny,
                                              params.nghost, params.nx_total);
    applyZeroGradientBC_Y<<<blocks_y, threads>>>(d_v, params.nx, params.ny,
                                                  params.nghost, params.nx_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif // BOUNDARY_CUH
