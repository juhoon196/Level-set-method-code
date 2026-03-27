/**
 * @file rk3.cuh
 * @brief 3rd-order TVD (Total Variation Diminishing) Runge-Kutta time integration
 *
 * This module implements the 3-stage 3rd-order TVD Runge-Kutta scheme for
 * time integration of the G-equation.
 *
 * The scheme is:
 *   Stage 1: G^(1) = G^n + dt * L(G^n)
 *   Stage 2: G^(2) = (3/4)*G^n + (1/4)*G^(1) + (1/4)*dt*L(G^(1))
 *   Stage 3: G^(n+1) = (1/3)*G^n + (2/3)*G^(2) + (2/3)*dt*L(G^(2))
 *
 * Reference: Shu & Osher (1988), "Efficient Implementation of Essentially
 *            Non-Oscillatory Shock-Capturing Schemes"
 */

#ifndef RK3_CUH
#define RK3_CUH

#include <cuda_runtime.h>
#include "config.cuh"
#include "weno5.cuh"

//=============================================================================
// RK3 Constants (Shu-Osher form)
//=============================================================================

// Stage 1: G^(1) = G^n + dt * L(G^n)
// alpha1 = 1.0, beta1 = 1.0

// Stage 2: G^(2) = (3/4)*G^n + (1/4)*G^(1) + (1/4)*dt*L(G^(1))
__constant__ double RK3_ALPHA2_0 = 0.75;   // Coefficient for G^n
__constant__ double RK3_ALPHA2_1 = 0.25;   // Coefficient for G^(1)
__constant__ double RK3_BETA2 = 0.25;      // Coefficient for dt*L

// Stage 3: G^(n+1) = (1/3)*G^n + (2/3)*G^(2) + (2/3)*dt*L(G^(2))
__constant__ double RK3_ALPHA3_0 = 1.0 / 3.0;  // Coefficient for G^n
__constant__ double RK3_ALPHA3_2 = 2.0 / 3.0;  // Coefficient for G^(2)
__constant__ double RK3_BETA3 = 2.0 / 3.0;     // Coefficient for dt*L

//=============================================================================
// RHS (Spatial Operator) Computation
//=============================================================================

/**
 * @brief Compute the right-hand side of the G-equation: -u_eff · ∇G
 *
 * The G-equation is: ∂G/∂t + u_eff · ∇G = 0
 * where u_eff = u - S_L * (∇G / |∇G|)
 *
 * @param G Level-set field
 * @param G_rhs Output RHS array
 * @param u Velocity field (x-component)
 * @param v Velocity field (y-component)
 * @param params Simulation parameters
 */
__global__ void computeRHS(const double* __restrict__ G,
                           double* __restrict__ G_rhs,
                           const double* __restrict__ u,
                           const double* __restrict__ v,
                           SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only compute for interior points
    if (i < params.nghost || i >= params.nx + params.nghost ||
        j < params.nghost || j >= params.ny + params.nghost) {
        return;
    }

    int index = idx(i, j, params.nx_total);

    // Get local velocity
    double u_local = u[index];
    double v_local = v[index];

    // Compute gradient for flame speed term if S_L > 0
    double dGdx_central = 0.0, dGdy_central = 0.0;
    double grad_mag = 1.0;  // Default to avoid division by zero

    if (params.s_l > params.epsilon) {
        // Central difference for gradient direction
        dGdx_central = (G[idx(i + 1, j, params.nx_total)] -
                        G[idx(i - 1, j, params.nx_total)]) / (2.0 * params.dx);
        dGdy_central = (G[idx(i, j + 1, params.nx_total)] -
                        G[idx(i, j - 1, params.nx_total)]) / (2.0 * params.dy);

        grad_mag = sqrt(dGdx_central * dGdx_central +
                        dGdy_central * dGdy_central + params.epsilon);
    }

    // Compute effective velocity: u_eff = u - S_L * (∇G / |∇G|)
    double u_eff = u_local - params.s_l * dGdx_central / grad_mag;
    double v_eff = v_local - params.s_l * dGdy_central / grad_mag;

    // Compute upwind derivatives using WENO-5
    double dGdx = weno5_dx(G, i, j, u_eff, params.dx, params.nx_total);
    double dGdy = weno5_dy(G, i, j, v_eff, params.dy, params.nx_total);

    // RHS = -u_eff · ∇G
    G_rhs[index] = -(u_eff * dGdx + v_eff * dGdy);
}

/**
 * @brief RK3 Stage 1: G^(1) = G^n + dt * L(G^n)
 */
__global__ void rk3Stage1(const double* __restrict__ G_n,
                          const double* __restrict__ G_rhs,
                          double* __restrict__ G_1,
                          double dt,
                          SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    int index = idx(i, j, params.nx_total);
    G_1[index] = G_n[index] + dt * G_rhs[index];
}

/**
 * @brief RK3 Stage 2: G^(2) = (3/4)*G^n + (1/4)*G^(1) + (1/4)*dt*L(G^(1))
 */
__global__ void rk3Stage2(const double* __restrict__ G_n,
                          const double* __restrict__ G_1,
                          const double* __restrict__ G_rhs,
                          double* __restrict__ G_2,
                          double dt,
                          SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    int index = idx(i, j, params.nx_total);
    G_2[index] = 0.75 * G_n[index] + 0.25 * G_1[index] + 0.25 * dt * G_rhs[index];
}

/**
 * @brief RK3 Stage 3: G^(n+1) = (1/3)*G^n + (2/3)*G^(2) + (2/3)*dt*L(G^(2))
 */
__global__ void rk3Stage3(const double* __restrict__ G_n,
                          const double* __restrict__ G_2,
                          const double* __restrict__ G_rhs,
                          double* __restrict__ G_new,
                          double dt,
                          SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    int index = idx(i, j, params.nx_total);
    double alpha0 = 1.0 / 3.0;
    double alpha2 = 2.0 / 3.0;
    double beta3 = 2.0 / 3.0;
    G_new[index] = alpha0 * G_n[index] + alpha2 * G_2[index] + beta3 * dt * G_rhs[index];
}

//=============================================================================
// Full RK3 Time Step (Host Function)
//=============================================================================

/**
 * @brief Perform one complete RK3 time step
 *
 * @param d_G_n Current solution (device)
 * @param d_G_new New solution (device)
 * @param d_G_1 Temporary storage for stage 1 (device)
 * @param d_G_2 Temporary storage for stage 2 (device)
 * @param d_G_rhs Temporary storage for RHS (device)
 * @param d_u Velocity field x-component (device)
 * @param d_v Velocity field y-component (device)
 * @param params Simulation parameters
 * @param applyBoundary Pointer to boundary condition function
 */
void rk3TimeStep(double* d_G_n, double* d_G_new,
                 double* d_G_1, double* d_G_2, double* d_G_rhs,
                 const double* d_u, const double* d_v,
                 SimParams& params,
                 void (*applyBoundary)(double*, SimParams&)) {

    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    // Stage 1: G^(1) = G^n + dt * L(G^n)
    computeRHS<<<grid, block>>>(d_G_n, d_G_rhs, d_u, d_v, params);
    CUDA_CHECK(cudaGetLastError());
    rk3Stage1<<<grid, block>>>(d_G_n, d_G_rhs, d_G_1, params.dt, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    applyBoundary(d_G_1, params);

    // Stage 2: G^(2) = (3/4)*G^n + (1/4)*G^(1) + (1/4)*dt*L(G^(1))
    computeRHS<<<grid, block>>>(d_G_1, d_G_rhs, d_u, d_v, params);
    CUDA_CHECK(cudaGetLastError());
    rk3Stage2<<<grid, block>>>(d_G_n, d_G_1, d_G_rhs, d_G_2, params.dt, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    applyBoundary(d_G_2, params);

    // Stage 3: G^(n+1) = (1/3)*G^n + (2/3)*G^(2) + (2/3)*dt*L(G^(2))
    computeRHS<<<grid, block>>>(d_G_2, d_G_rhs, d_u, d_v, params);
    CUDA_CHECK(cudaGetLastError());
    rk3Stage3<<<grid, block>>>(d_G_n, d_G_2, d_G_rhs, d_G_new, params.dt, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    applyBoundary(d_G_new, params);
}

#endif // RK3_CUH
