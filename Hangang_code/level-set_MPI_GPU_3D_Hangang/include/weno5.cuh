/**
 * @file weno5_3d.cuh
 * @brief 5th-order WENO (Weighted Essentially Non-Oscillatory) scheme implementation for 3D
 *
 * This module implements the WENO-5 scheme for spatial discretization of the
 * convective term in the G-equation for 3D problems.
 *
 * Reference: Jiang & Shu (1996), "Efficient Implementation of Weighted ENO Schemes"
 */

#ifndef WENO5_CUH
#define WENO5_CUH

#include <cuda_runtime.h>
#include "config.cuh"

//=============================================================================
// WENO-5 Constants
//=============================================================================

// Optimal weights for smooth regions (5th-order accuracy)
__constant__ double WENO_D0 = 0.1;
__constant__ double WENO_D1 = 0.6;
__constant__ double WENO_D2 = 0.3;

// Epsilon for smoothness indicator (prevents division by zero)
__constant__ double WENO_EPSILON = 1.0e-6;

//=============================================================================
// WENO-5 Device Functions (reuse 1D reconstruction)
//=============================================================================

/**
 * @brief Compute smoothness indicators for WENO-5
 * @param v Array of 5 stencil values (v[0] to v[4])
 * @param beta Output array of 3 smoothness indicators
 */
__device__ inline void computeSmoothnessIndicators(const double* v, double* beta) {
    // Smoothness indicator for stencil 0: {v[0], v[1], v[2]}
    double d1_0 = v[1] - v[0];
    double d2_0 = v[2] - 2.0 * v[1] + v[0];
    beta[0] = (13.0 / 12.0) * d2_0 * d2_0 + 0.25 * (v[2] - v[0]) * (v[2] - v[0]);

    // Smoothness indicator for stencil 1: {v[1], v[2], v[3]}
    double d1_1 = v[2] - v[1];
    double d2_1 = v[3] - 2.0 * v[2] + v[1];
    beta[1] = (13.0 / 12.0) * d2_1 * d2_1 + 0.25 * (v[3] - v[1]) * (v[3] - v[1]);

    // Smoothness indicator for stencil 2: {v[2], v[3], v[4]}
    double d1_2 = v[3] - v[2];
    double d2_2 = v[4] - 2.0 * v[3] + v[2];
    beta[2] = (13.0 / 12.0) * d2_2 * d2_2 + 0.25 * (v[4] - v[2]) * (v[4] - v[2]);
}

/**
 * @brief Compute WENO-5 reconstruction at cell interface (left-biased)
 */
__device__ inline double weno5_left(const double* v) {
    // Smoothness indicators
    double beta[3];
    computeSmoothnessIndicators(v, beta);

    // Compute nonlinear weights
    double alpha0 = 0.1 / ((WENO_EPSILON + beta[0]) * (WENO_EPSILON + beta[0]));
    double alpha1 = 0.6 / ((WENO_EPSILON + beta[1]) * (WENO_EPSILON + beta[1]));
    double alpha2 = 0.3 / ((WENO_EPSILON + beta[2]) * (WENO_EPSILON + beta[2]));

    double sum_alpha = alpha0 + alpha1 + alpha2;

    double omega0 = alpha0 / sum_alpha;
    double omega1 = alpha1 / sum_alpha;
    double omega2 = alpha2 / sum_alpha;

    // Candidate polynomials
    double p0 = (2.0 * v[0] - 7.0 * v[1] + 11.0 * v[2]) / 6.0;
    double p1 = (-v[1] + 5.0 * v[2] + 2.0 * v[3]) / 6.0;
    double p2 = (2.0 * v[2] + 5.0 * v[3] - v[4]) / 6.0;

    // Weighted combination
    return omega0 * p0 + omega1 * p1 + omega2 * p2;
}

/**
 * @brief Compute WENO-5 reconstruction at cell interface (right-biased)
 */
__device__ inline double weno5_right(const double* v) {
    // Mirror the stencil for right-biased reconstruction
    double v_mirror[5] = {v[4], v[3], v[2], v[1], v[0]};

    // Smoothness indicators
    double beta[3];
    computeSmoothnessIndicators(v_mirror, beta);

    // Compute nonlinear weights with mirrored optimal weights
    double alpha0 = 0.3 / ((WENO_EPSILON + beta[0]) * (WENO_EPSILON + beta[0]));
    double alpha1 = 0.6 / ((WENO_EPSILON + beta[1]) * (WENO_EPSILON + beta[1]));
    double alpha2 = 0.1 / ((WENO_EPSILON + beta[2]) * (WENO_EPSILON + beta[2]));

    double sum_alpha = alpha0 + alpha1 + alpha2;

    double omega0 = alpha0 / sum_alpha;
    double omega1 = alpha1 / sum_alpha;
    double omega2 = alpha2 / sum_alpha;

    // Candidate polynomials for right reconstruction
    double p0 = (11.0 * v[2] - 7.0 * v[3] + 2.0 * v[4]) / 6.0;
    double p1 = (2.0 * v[1] + 5.0 * v[2] - v[3]) / 6.0;
    double p2 = (-v[0] + 5.0 * v[1] + 2.0 * v[2]) / 6.0;

    // Weighted combination
    return omega0 * p0 + omega1 * p1 + omega2 * p2;
}

//=============================================================================
// 3D WENO-5 Derivatives
//=============================================================================

/**
 * @brief Compute upwind derivative using WENO-5 in x-direction (3D)
 */
__device__ inline double weno5_dx(const double* G, int i, int j, int k,
                                      double u_eff, double dx,
                                      int nx_total, int ny_total) {
    // Gather stencil values
    double v[5];
    v[0] = G[idx(i - 2, j, k, nx_total, ny_total)];
    v[1] = G[idx(i - 1, j, k, nx_total, ny_total)];
    v[2] = G[idx(i, j, k, nx_total, ny_total)];
    v[3] = G[idx(i + 1, j, k, nx_total, ny_total)];
    v[4] = G[idx(i + 2, j, k, nx_total, ny_total)];

    if (u_eff >= 0.0) {
        // Upwind from left
        double v_left[5] = {
            G[idx(i - 3, j, k, nx_total, ny_total)],
            G[idx(i - 2, j, k, nx_total, ny_total)],
            G[idx(i - 1, j, k, nx_total, ny_total)],
            G[idx(i, j, k, nx_total, ny_total)],
            G[idx(i + 1, j, k, nx_total, ny_total)]
        };
        double G_left = weno5_left(v_left);
        double G_right = weno5_left(v);
        return (G_right - G_left) / dx;
    } else {
        // Upwind from right
        double v_right[5] = {
            G[idx(i - 1, j, k, nx_total, ny_total)],
            G[idx(i, j, k, nx_total, ny_total)],
            G[idx(i + 1, j, k, nx_total, ny_total)],
            G[idx(i + 2, j, k, nx_total, ny_total)],
            G[idx(i + 3, j, k, nx_total, ny_total)]
        };
        double G_left = weno5_right(v);
        double G_right = weno5_right(v_right);
        return (G_right - G_left) / dx;
    }
}

/**
 * @brief Compute upwind derivative using WENO-5 in y-direction (3D)
 */
__device__ inline double weno5_dy(const double* G, int i, int j, int k,
                                      double v_eff, double dy,
                                      int nx_total, int ny_total) {
    // Gather stencil values
    double v[5];
    v[0] = G[idx(i, j - 2, k, nx_total, ny_total)];
    v[1] = G[idx(i, j - 1, k, nx_total, ny_total)];
    v[2] = G[idx(i, j, k, nx_total, ny_total)];
    v[3] = G[idx(i, j + 1, k, nx_total, ny_total)];
    v[4] = G[idx(i, j + 2, k, nx_total, ny_total)];

    if (v_eff >= 0.0) {
        // Upwind from bottom
        double v_bottom[5] = {
            G[idx(i, j - 3, k, nx_total, ny_total)],
            G[idx(i, j - 2, k, nx_total, ny_total)],
            G[idx(i, j - 1, k, nx_total, ny_total)],
            G[idx(i, j, k, nx_total, ny_total)],
            G[idx(i, j + 1, k, nx_total, ny_total)]
        };
        double G_bottom = weno5_left(v_bottom);
        double G_top = weno5_left(v);
        return (G_top - G_bottom) / dy;
    } else {
        // Upwind from top
        double v_top[5] = {
            G[idx(i, j - 1, k, nx_total, ny_total)],
            G[idx(i, j, k, nx_total, ny_total)],
            G[idx(i, j + 1, k, nx_total, ny_total)],
            G[idx(i, j + 2, k, nx_total, ny_total)],
            G[idx(i, j + 3, k, nx_total, ny_total)]
        };
        double G_bottom = weno5_right(v);
        double G_top = weno5_right(v_top);
        return (G_top - G_bottom) / dy;
    }
}

/**
 * @brief Compute upwind derivative using WENO-5 in z-direction (3D)
 */
__device__ inline double weno5_dz(const double* G, int i, int j, int k,
                                      double w_eff, double dz,
                                      int nx_total, int ny_total) {
    // Gather stencil values
    double v[5];
    v[0] = G[idx(i, j, k - 2, nx_total, ny_total)];
    v[1] = G[idx(i, j, k - 1, nx_total, ny_total)];
    v[2] = G[idx(i, j, k, nx_total, ny_total)];
    v[3] = G[idx(i, j, k + 1, nx_total, ny_total)];
    v[4] = G[idx(i, j, k + 2, nx_total, ny_total)];

    if (w_eff >= 0.0) {
        // Upwind from back
        double v_back[5] = {
            G[idx(i, j, k - 3, nx_total, ny_total)],
            G[idx(i, j, k - 2, nx_total, ny_total)],
            G[idx(i, j, k - 1, nx_total, ny_total)],
            G[idx(i, j, k, nx_total, ny_total)],
            G[idx(i, j, k + 1, nx_total, ny_total)]
        };
        double G_back = weno5_left(v_back);
        double G_front = weno5_left(v);
        return (G_front - G_back) / dz;
    } else {
        // Upwind from front
        double v_front[5] = {
            G[idx(i, j, k - 1, nx_total, ny_total)],
            G[idx(i, j, k, nx_total, ny_total)],
            G[idx(i, j, k + 1, nx_total, ny_total)],
            G[idx(i, j, k + 2, nx_total, ny_total)],
            G[idx(i, j, k + 3, nx_total, ny_total)]
        };
        double G_back = weno5_right(v);
        double G_front = weno5_right(v_front);
        return (G_front - G_back) / dz;
    }
}

/**
 * @brief Compute gradient magnitude using WENO-5 (for reinitialization) - 3D version
 */
__device__ inline double weno5_gradient_magnitude(const double* G, int i, int j, int k,
                                                      double sign, double dx, double dy, double dz,
                                                      int nx_total, int ny_total) {
    // Compute one-sided derivatives in x-direction
    double dG_dx_minus = (G[idx(i, j, k, nx_total, ny_total)] -
                          G[idx(i - 1, j, k, nx_total, ny_total)]) / dx;
    double dG_dx_plus = (G[idx(i + 1, j, k, nx_total, ny_total)] -
                         G[idx(i, j, k, nx_total, ny_total)]) / dx;

    // Compute one-sided derivatives in y-direction
    double dG_dy_minus = (G[idx(i, j, k, nx_total, ny_total)] -
                          G[idx(i, j - 1, k, nx_total, ny_total)]) / dy;
    double dG_dy_plus = (G[idx(i, j + 1, k, nx_total, ny_total)] -
                         G[idx(i, j, k, nx_total, ny_total)]) / dy;

    // Compute one-sided derivatives in z-direction
    double dG_dz_minus = (G[idx(i, j, k, nx_total, ny_total)] -
                          G[idx(i, j, k - 1, nx_total, ny_total)]) / dz;
    double dG_dz_plus = (G[idx(i, j, k + 1, nx_total, ny_total)] -
                         G[idx(i, j, k, nx_total, ny_total)]) / dz;

    // Godunov scheme for Hamilton-Jacobi equations
    double grad_mag_sq;

    if (sign > 0.0) {
        // Expanding interface (phi > 0): use min of |grad|
        double ax = fmax(dG_dx_minus, 0.0);
        double bx = fmin(dG_dx_plus, 0.0);
        double ay = fmax(dG_dy_minus, 0.0);
        double by = fmin(dG_dy_plus, 0.0);
        double az = fmax(dG_dz_minus, 0.0);
        double bz = fmin(dG_dz_plus, 0.0);

        grad_mag_sq = fmax(ax * ax, bx * bx) + fmax(ay * ay, by * by) + fmax(az * az, bz * bz);
    } else {
        // Contracting interface (phi < 0): use max of |grad|
        double ax = fmin(dG_dx_minus, 0.0);
        double bx = fmax(dG_dx_plus, 0.0);
        double ay = fmin(dG_dy_minus, 0.0);
        double by = fmax(dG_dy_plus, 0.0);
        double az = fmin(dG_dz_minus, 0.0);
        double bz = fmax(dG_dz_plus, 0.0);

        grad_mag_sq = fmax(ax * ax, bx * bx) + fmax(ay * ay, by * by) + fmax(az * az, bz * bz);
    }

    return sqrt(grad_mag_sq);
}

/**
 * @brief Compute WENO-5 gradient (central differences fallback) - 3D version
 */
__device__ inline void weno5_gradient(const double* G, int i, int j, int k,
                                          double dx, double dy, double dz,
                                          int nx_total, int ny_total,
                                          double& dGdx, double& dGdy, double& dGdz) {
    // Simple central differences for now (can be improved with WENO)
    dGdx = (G[idx(i + 1, j, k, nx_total, ny_total)] -
            G[idx(i - 1, j, k, nx_total, ny_total)]) / (2.0 * dx);
    dGdy = (G[idx(i, j + 1, k, nx_total, ny_total)] -
            G[idx(i, j - 1, k, nx_total, ny_total)]) / (2.0 * dy);
    dGdz = (G[idx(i, j, k + 1, nx_total, ny_total)] -
            G[idx(i, j, k - 1, nx_total, ny_total)]) / (2.0 * dz);
}

#endif // WENO5_CUH
