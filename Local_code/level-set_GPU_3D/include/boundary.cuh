/**
 * @file boundary_3d.cuh
 * @brief Boundary condition implementations for 3D G-equation solver
 *
 * This module provides periodic boundary conditions in all directions
 * for the 3D deformation test.
 */

#ifndef BOUNDARY_CUH
#define BOUNDARY_CUH

#include <cuda_runtime.h>
#include "config.cuh"

//=============================================================================
// Periodic Boundary Conditions (3D)
//=============================================================================

/**
 * @brief Apply periodic BC in x-direction (3D)
 */
__global__ void applyPeriodicBC_X_3D(double* G, int nx, int ny, int nz, int nghost,
                                      int nx_total, int ny_total) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= ny + 2 * nghost || k >= nz + 2 * nghost) return;

    // Left ghost cells
    for (int g = 0; g < nghost; g++) {
        int src_i = nx + g;
        int dst_i = g;
        G[idx(dst_i, j, k, nx_total, ny_total)] = G[idx(src_i, j, k, nx_total, ny_total)];
    }

    // Right ghost cells
    for (int g = 0; g < nghost; g++) {
        int src_i = nghost + g;
        int dst_i = nx + nghost + g;
        G[idx(dst_i, j, k, nx_total, ny_total)] = G[idx(src_i, j, k, nx_total, ny_total)];
    }
}

/**
 * @brief Apply periodic BC in y-direction (3D)
 */
__global__ void applyPeriodicBC_Y_3D(double* G, int nx, int ny, int nz, int nghost,
                                      int nx_total, int ny_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx + 2 * nghost || k >= nz + 2 * nghost) return;

    // Bottom ghost cells
    for (int g = 0; g < nghost; g++) {
        int src_j = ny + g;
        int dst_j = g;
        G[idx(i, dst_j, k, nx_total, ny_total)] = G[idx(i, src_j, k, nx_total, ny_total)];
    }

    // Top ghost cells
    for (int g = 0; g < nghost; g++) {
        int src_j = nghost + g;
        int dst_j = ny + nghost + g;
        G[idx(i, dst_j, k, nx_total, ny_total)] = G[idx(i, src_j, k, nx_total, ny_total)];
    }
}

/**
 * @brief Apply periodic BC in z-direction (3D)
 */
__global__ void applyPeriodicBC_Z_3D(double* G, int nx, int ny, int nz, int nghost,
                                      int nx_total, int ny_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx + 2 * nghost || j >= ny + 2 * nghost) return;

    // Back ghost cells
    for (int g = 0; g < nghost; g++) {
        int src_k = nz + g;
        int dst_k = g;
        G[idx(i, j, dst_k, nx_total, ny_total)] = G[idx(i, j, src_k, nx_total, ny_total)];
    }

    // Front ghost cells
    for (int g = 0; g < nghost; g++) {
        int src_k = nghost + g;
        int dst_k = nz + nghost + g;
        G[idx(i, j, dst_k, nx_total, ny_total)] = G[idx(i, j, src_k, nx_total, ny_total)];
    }
}

//=============================================================================
// Combined Boundary Condition Application
//=============================================================================

/**
 * @brief Apply periodic boundary conditions in all directions (3D)
 *
 * For the deformation test, we use periodic BCs in all directions.
 */
void applyBoundaryConditions(double* d_G, SimParams& params) {
    int threads_2d = 16;
    dim3 block_2d(threads_2d, threads_2d);

    // X-direction BC (operates on YZ planes)
    int grid_y = (params.ny_total + threads_2d - 1) / threads_2d;
    int grid_z = (params.nz_total + threads_2d - 1) / threads_2d;
    dim3 grid_x(grid_y, grid_z);
    applyPeriodicBC_X_3D<<<grid_x, block_2d>>>(d_G, params.nx, params.ny, params.nz,
                                                params.nghost, params.nx_total, params.ny_total);

    // Y-direction BC (operates on XZ planes)
    int grid_x_dim = (params.nx_total + threads_2d - 1) / threads_2d;
    dim3 grid_y_dim(grid_x_dim, grid_z);
    applyPeriodicBC_Y_3D<<<grid_y_dim, block_2d>>>(d_G, params.nx, params.ny, params.nz,
                                                     params.nghost, params.nx_total, params.ny_total);

    // Z-direction BC (operates on XY planes)
    dim3 grid_z_dim(grid_x_dim, grid_y);
    applyPeriodicBC_Z_3D<<<grid_z_dim, block_2d>>>(d_G, params.nx, params.ny, params.nz,
                                                     params.nghost, params.nx_total, params.ny_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Apply boundary conditions to velocity field components (3D)
 */
void applyVelocityBoundaryConditions(double* d_u, double* d_v, double* d_w, SimParams& params) {
    // Apply periodic BCs to all velocity components
    applyBoundaryConditions(d_u, params);
    applyBoundaryConditions(d_v, params);
    applyBoundaryConditions(d_w, params);
}

#endif // BOUNDARY_CUH
