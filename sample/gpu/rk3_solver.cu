#include "rk3_solver.cuh"
#include <cmath>

// ============================================================
// RK3 Stage 1: G1 = G + dt * rhs
// ============================================================

__global__ void kernel_rk3_stage1(
    const double* __restrict__ G,
    const double* __restrict__ rhs,
    double* __restrict__ G1,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + GHOST;
    int j = blockIdx.y * blockDim.y + threadIdx.y + GHOST;

    if (i > NX + GHOST - 1 || j > NY + GHOST - 1) return;

    int idx = idx2d(j, i);
    G1[idx] = G[idx] + dt * rhs[idx];
}

// ============================================================
// RK3 Stage 2: G2 = 0.75*G + 0.25*(G1 + dt*rhs)
// ============================================================

__global__ void kernel_rk3_stage2(
    const double* __restrict__ G,
    const double* __restrict__ G1,
    const double* __restrict__ rhs,
    double* __restrict__ G2,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + GHOST;
    int j = blockIdx.y * blockDim.y + threadIdx.y + GHOST;

    if (i > NX + GHOST - 1 || j > NY + GHOST - 1) return;

    int idx = idx2d(j, i);
    G2[idx] = 0.75 * G[idx] + 0.25 * (G1[idx] + dt * rhs[idx]);
}

// ============================================================
// RK3 Stage 3: G = (1/3)*G + (2/3)*(G2 + dt*rhs)
// ============================================================

__global__ void kernel_rk3_stage3(
    double* __restrict__ G,
    const double* __restrict__ G2,
    const double* __restrict__ rhs,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + GHOST;
    int j = blockIdx.y * blockDim.y + threadIdx.y + GHOST;

    if (i > NX + GHOST - 1 || j > NY + GHOST - 1) return;

    int idx = idx2d(j, i);
    G[idx] = (1.0 / 3.0) * G[idx] + (2.0 / 3.0) * (G2[idx] + dt * rhs[idx]);
}

// ============================================================
// Velocity update kernel
// ============================================================

__global__ void kernel_update_velocity(
    double* __restrict__ u,
    double* __restrict__ v,
    double t,
    int freq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX_TOTAL || j >= NY_TOTAL) return;

    int idx = idx2d(j, i);
    double omega = freq * 2.0 * M_PI;
    double cos_val = cos(omega * t);

    u[idx] = u0 + epsilon_u * v0 * cos_val;
    v[idx] = v0 + epsilon_v * v0 * cos_val;
}

// ============================================================
// Complete RK3 step wrapper
// ============================================================

void RK3_step_gpu(
    double* d_G,
    double* d_G1,
    double* d_G2,
    const double* d_u,
    const double* d_v,
    double* d_rhs,
    const double* d_G_initial,
    double dx,
    double dy,
    double dt,
    double S_L
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((NX + block.x - 1) / block.x, (NY + block.y - 1) / block.y);

    // Stage 1
    compute_geqn_rhs_gpu(d_G, d_u, d_v, d_rhs, dx, dy, S_L);
    kernel_rk3_stage1<<<grid, block>>>(d_G, d_rhs, d_G1, dt);
    CUDA_CHECK(cudaGetLastError());
    apply_anchored_bc_gpu(d_G1, d_G_initial);
    apply_periodic_gpu(d_G1);

    // Stage 2
    compute_geqn_rhs_gpu(d_G1, d_u, d_v, d_rhs, dx, dy, S_L);
    kernel_rk3_stage2<<<grid, block>>>(d_G, d_G1, d_rhs, d_G2, dt);
    CUDA_CHECK(cudaGetLastError());
    apply_anchored_bc_gpu(d_G2, d_G_initial);
    apply_periodic_gpu(d_G2);

    // Stage 3
    compute_geqn_rhs_gpu(d_G2, d_u, d_v, d_rhs, dx, dy, S_L);
    kernel_rk3_stage3<<<grid, block>>>(d_G, d_G2, d_rhs, dt);
    CUDA_CHECK(cudaGetLastError());
    apply_anchored_bc_gpu(d_G, d_G_initial);
    apply_periodic_gpu(d_G);
}
