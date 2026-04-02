/**
 * @file rk3.cuh
 * @brief 3rd-order TVD Runge-Kutta time integration (MPI + CUDA)
 *
 * Identical numerics to single-GPU version.
 * Only changes: getGridDim(params), boundary function takes HaloBuffers.
 */

#ifndef RK3_CUH
#define RK3_CUH

#include <cuda_runtime.h>
#include "config.cuh"
#include "weno5.cuh"
#include "boundary.cuh"

//=============================================================================
// RHS Computation
//=============================================================================

__global__ void computeRHS(const double* __restrict__ G,
                              double* __restrict__ G_rhs,
                              const double* __restrict__ u,
                              const double* __restrict__ v,
                              const double* __restrict__ w,
                              SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Interior only (params.ny is local_ny)
    if (i < params.nghost || i >= params.nx + params.nghost ||
        j < params.nghost || j >= params.ny + params.nghost ||
        k < params.nghost || k >= params.nz + params.nghost)
        return;

    int index = idx(i, j, k, params.nx_total, params.ny_total);
    double u_eff{u[index]};
    double v_eff{v[index]};
    double w_eff{w[index]};

    const double eps{params.epsilon};
    if (params.s_l > eps) {
        double dGdx_c = (G[idx(i+1,j,k,params.nx_total,params.ny_total)] -
                         G[idx(i-1,j,k,params.nx_total,params.ny_total)]) / (2.0*params.dx);
        double dGdy_c = (G[idx(i,j+1,k,params.nx_total,params.ny_total)] -
                         G[idx(i,j-1,k,params.nx_total,params.ny_total)]) / (2.0*params.dy);
        double dGdz_c = (G[idx(i,j,k+1,params.nx_total,params.ny_total)] -
                         G[idx(i,j,k-1,params.nx_total,params.ny_total)]) / (2.0*params.dz);

        double grad_mag_inv{rsqrt(eps + dGdx_c*dGdx_c + dGdy_c*dGdy_c + dGdz_c*dGdz_c)};
        u_eff = u_eff - (params.s_l * dGdx_c) * grad_mag_inv;
        v_eff = v_eff - (params.s_l * dGdy_c) * grad_mag_inv;
        w_eff = w_eff - (params.s_l * dGdz_c) * grad_mag_inv;
    }

    double dGdx = weno5_dx(G, i, j, k, u_eff, params.dx, params.nx_total, params.ny_total);
    double dGdy = weno5_dy(G, i, j, k, v_eff, params.dy, params.nx_total, params.ny_total);
    double dGdz = weno5_dz(G, i, j, k, w_eff, params.dz, params.nx_total, params.ny_total);

    G_rhs[index] = -(u_eff * dGdx + v_eff * dGdy + w_eff * dGdz);
}

//=============================================================================
// RK3 Stage Kernels
//=============================================================================

__global__ void rk3Stage1(const double* __restrict__ G_n,
                              const double* __restrict__ G_rhs,
                              double* __restrict__ G_1,
                              double dt, SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= params.nx_total || j >= params.ny_total || k >= params.nz_total) return;

    int index = idx(i, j, k, params.nx_total, params.ny_total);
    G_1[index] = G_n[index] + dt * G_rhs[index];
}

__global__ void rk3Stage2(const double* __restrict__ G_n,
                              const double* __restrict__ G_1,
                              const double* __restrict__ G_rhs,
                              double* __restrict__ G_2,
                              double dt, SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= params.nx_total || j >= params.ny_total || k >= params.nz_total) return;

    int index = idx(i, j, k, params.nx_total, params.ny_total);
    G_2[index] = 0.75 * G_n[index] + 0.25 * G_1[index] + 0.25 * dt * G_rhs[index];
}

__global__ void rk3Stage3(const double* __restrict__ G_n,
                              const double* __restrict__ G_2,
                              const double* __restrict__ G_rhs,
                              double* __restrict__ G_new,
                              double dt, SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= params.nx_total || j >= params.ny_total || k >= params.nz_total) return;

    int index = idx(i, j, k, params.nx_total, params.ny_total);
    G_new[index] = (1.0/3.0) * G_n[index] + (2.0/3.0) * G_2[index]
                 + (2.0/3.0) * dt * G_rhs[index];
}

//=============================================================================
// Full RK3 Time Step
//=============================================================================

void rk3TimeStep(double* d_G_n, double* d_G_new,
                   double* d_G_1, double* d_G_2, double* d_G_rhs,
                   const double* d_u, const double* d_v, const double* d_w,
                   SimParams& params, HaloBuffers& halo) {

    dim3 grid = getGridDim(params);
    dim3 block = getBlockDim();

    // Stage 1: G^(1) = G^n + dt * L(G^n)
    computeRHS<<<grid, block>>>(d_G_n, d_G_rhs, d_u, d_v, d_w, params);
    CUDA_CHECK(cudaGetLastError());
    rk3Stage1<<<grid, block>>>(d_G_n, d_G_rhs, d_G_1, params.dt, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    applyBoundaryConditions(d_G_1, params, halo);

    // Stage 2: G^(2) = 3/4 G^n + 1/4 G^(1) + 1/4 dt L(G^(1))
    computeRHS<<<grid, block>>>(d_G_1, d_G_rhs, d_u, d_v, d_w, params);
    CUDA_CHECK(cudaGetLastError());
    rk3Stage2<<<grid, block>>>(d_G_n, d_G_1, d_G_rhs, d_G_2, params.dt, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    applyBoundaryConditions(d_G_2, params, halo);

    // Stage 3: G^(n+1) = 1/3 G^n + 2/3 G^(2) + 2/3 dt L(G^(2))
    computeRHS<<<grid, block>>>(d_G_2, d_G_rhs, d_u, d_v, d_w, params);
    CUDA_CHECK(cudaGetLastError());
    rk3Stage3<<<grid, block>>>(d_G_n, d_G_2, d_G_rhs, d_G_new, params.dt, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    applyBoundaryConditions(d_G_new, params, halo);
}

#endif // RK3_CUH
