#include "weno5_kernels.cuh"
#include <cmath>

// ============================================================
// WENO5 helper device functions
// ============================================================

__device__ inline double beta_weno(double f0, double f1, double f2) {
    double d1 = f0 - 2.0 * f1 + f2;
    double d2 = f0 - 4.0 * f1 + 3.0 * f2;
    return (13.0 / 12.0) * d1 * d1 + (1.0 / 4.0) * d2 * d2;
}

__device__ inline double weno5_left(double v0, double v1, double v2, double v3, double v4) {
    double b0 = beta_weno(v0, v1, v2);
    double b1 = beta_weno(v1, v2, v3);
    double b2 = beta_weno(v2, v3, v4);

    constexpr double g0 = 0.1, g1 = 0.6, g2 = 0.3;
    double eps = WENO_EPSILON;

    double a0 = g0 / ((eps + b0) * (eps + b0));
    double a1 = g1 / ((eps + b1) * (eps + b1));
    double a2 = g2 / ((eps + b2) * (eps + b2));
    double sum = a0 + a1 + a2;
    double w0 = a0 / sum;
    double w1 = a1 / sum;
    double w2 = a2 / sum;

    return w0 * (2.0 * v0 - 7.0 * v1 + 11.0 * v2) / 6.0
         + w1 * (-v1 + 5.0 * v2 + 2.0 * v3) / 6.0
         + w2 * (2.0 * v2 + 5.0 * v3 - v4) / 6.0;
}

__device__ inline double weno5_right(double v0, double v1, double v2, double v3, double v4) {
    double b0 = beta_weno(v4, v3, v2);
    double b1 = beta_weno(v3, v2, v1);
    double b2 = beta_weno(v2, v1, v0);

    constexpr double g0 = 0.1, g1 = 0.6, g2 = 0.3;
    double eps = WENO_EPSILON;

    double a0 = g0 / ((eps + b0) * (eps + b0));
    double a1 = g1 / ((eps + b1) * (eps + b1));
    double a2 = g2 / ((eps + b2) * (eps + b2));
    double sum = a0 + a1 + a2;
    double w0 = a0 / sum;
    double w1 = a1 / sum;
    double w2 = a2 / sum;

    return w0 * (2.0 * v4 - 7.0 * v3 + 11.0 * v2) / 6.0
         + w1 * (-v3 + 5.0 * v2 + 2.0 * v1) / 6.0
         + w2 * (2.0 * v2 + 5.0 * v1 - v0) / 6.0;
}

// ============================================================
// Combined kernel: compute RHS with inline WENO5
// ============================================================

__global__ void kernel_compute_rhs_weno5(
    const double* __restrict__ G,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ rhs,
    double dx,
    double dy,
    double S_L
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + GHOST;
    int j = blockIdx.y * blockDim.y + threadIdx.y + GHOST;

    if (i > NX + GHOST - 1 || j > NY + GHOST - 1) return;

    int idx = idx2d(j, i);

    // X-direction WENO5 derivatives
    double dGdx_L = (weno5_left(G[idx2d(j, i-2)], G[idx2d(j, i-1)], G[idx2d(j, i)], G[idx2d(j, i+1)], G[idx2d(j, i+2)])
                   - weno5_left(G[idx2d(j, i-3)], G[idx2d(j, i-2)], G[idx2d(j, i-1)], G[idx2d(j, i)], G[idx2d(j, i+1)])) / dx;
    double dGdx_R = (weno5_right(G[idx2d(j, i+3)], G[idx2d(j, i+2)], G[idx2d(j, i+1)], G[idx2d(j, i)], G[idx2d(j, i-1)])
                   - weno5_right(G[idx2d(j, i+2)], G[idx2d(j, i+1)], G[idx2d(j, i)], G[idx2d(j, i-1)], G[idx2d(j, i-2)])) / dx;

    // Y-direction WENO5 derivatives
    double dGdy_L = (weno5_left(G[idx2d(j-2, i)], G[idx2d(j-1, i)], G[idx2d(j, i)], G[idx2d(j+1, i)], G[idx2d(j+2, i)])
                   - weno5_left(G[idx2d(j-3, i)], G[idx2d(j-2, i)], G[idx2d(j-1, i)], G[idx2d(j, i)], G[idx2d(j+1, i)])) / dy;
    double dGdy_R = (weno5_right(G[idx2d(j+3, i)], G[idx2d(j+2, i)], G[idx2d(j+1, i)], G[idx2d(j, i)], G[idx2d(j-1, i)])
                   - weno5_right(G[idx2d(j+2, i)], G[idx2d(j+1, i)], G[idx2d(j, i)], G[idx2d(j-1, i)], G[idx2d(j-2, i)])) / dy;

    // Mean gradients
    double Gdx_mean = 0.5 * (dGdx_L + dGdx_R);
    double Gdy_mean = 0.5 * (dGdy_L + dGdy_R);

    // Transport and propagation
    double u_val = u[idx];
    double v_val = v[idx];
    double transport = u_val * Gdx_mean + v_val * Gdy_mean;
    double grad_mag = sqrt(Gdx_mean * Gdx_mean + Gdy_mean * Gdy_mean);
    double propagation = S_L * grad_mag;

    // Local Lax-Friedrichs dissipation
    double grad_L = sqrt(dGdx_L * dGdx_L + dGdy_L * dGdy_L + 1e-14);
    double grad_R = sqrt(dGdx_R * dGdx_R + dGdy_R * dGdy_R + 1e-14);

    double s_xm = u_val - S_L * dGdx_L / grad_L;
    double s_xp = u_val - S_L * dGdx_R / grad_R;
    double alpha_x = fmax(fabs(s_xm), fabs(s_xp));

    double s_ym = v_val - S_L * dGdy_L / grad_L;
    double s_yp = v_val - S_L * dGdy_R / grad_R;
    double alpha_y = fmax(fabs(s_ym), fabs(s_yp));

    double alpha = fmax(alpha_x, alpha_y);

    double diss_x = alpha * (dGdx_R - dGdx_L) * 0.5;
    double diss_y = alpha * (dGdy_R - dGdy_L) * 0.5;

    // Hamiltonian
    double H = transport - propagation - (diss_x + diss_y);

    rhs[idx] = -H;
}

// ============================================================
// Wrapper function
// ============================================================

void compute_geqn_rhs_gpu(
    const double* d_G,
    const double* d_u,
    const double* d_v,
    double* d_rhs,
    double dx,
    double dy,
    double S_L
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((NX + block.x - 1) / block.x, (NY + block.y - 1) / block.y);

    kernel_compute_rhs_weno5<<<grid, block>>>(d_G, d_u, d_v, d_rhs, dx, dy, S_L);
    CUDA_CHECK(cudaGetLastError());
}
