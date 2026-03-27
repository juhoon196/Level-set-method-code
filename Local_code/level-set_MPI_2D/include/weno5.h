/**
 * @file weno5.h
 * @brief 5th-order WENO (Weighted Essentially Non-Oscillatory) scheme implementation
 *
 * This module implements the WENO-5 scheme for spatial discretization of the
 * convective term in the G-equation. The scheme provides high-order accuracy
 * while maintaining stability near discontinuities.
 *
 * Reference: Jiang & Shu (1996), "Efficient Implementation of Weighted ENO Schemes"
 */

#ifndef WENO5_H
#define WENO5_H

#include <cmath>
#include "config.h"

//=============================================================================
// WENO-5 Constants
//=============================================================================

// Optimal weights for smooth regions (5th-order accuracy)
constexpr double WENO_D0 = 0.1;
constexpr double WENO_D1 = 0.6;
constexpr double WENO_D2 = 0.3;

// Epsilon for smoothness indicator (prevents division by zero)
constexpr double WENO_EPSILON = 1.0e-6;

//=============================================================================
// WENO-5 Functions
//=============================================================================

/**
 * @brief Compute smoothness indicators for WENO-5
 * @param v Array of 5 stencil values (v[0] to v[4])
 * @param beta Output array of 3 smoothness indicators
 */
inline void computeSmoothnessIndicators(const double* v, double* beta) {
    // Smoothness indicator for stencil 0: {v[0], v[1], v[2]}
    double d2_0 = v[2] - 2.0 * v[1] + v[0];
    beta[0] = (13.0 / 12.0) * d2_0 * d2_0 + 0.25 * (v[2] - v[0]) * (v[2] - v[0]);

    // Smoothness indicator for stencil 1: {v[1], v[2], v[3]}
    double d2_1 = v[3] - 2.0 * v[2] + v[1];
    beta[1] = (13.0 / 12.0) * d2_1 * d2_1 + 0.25 * (v[3] - v[1]) * (v[3] - v[1]);

    // Smoothness indicator for stencil 2: {v[2], v[3], v[4]}
    double d2_2 = v[4] - 2.0 * v[3] + v[2];
    beta[2] = (13.0 / 12.0) * d2_2 * d2_2 + 0.25 * (v[4] - v[2]) * (v[4] - v[2]);
}

/**
 * @brief Compute WENO-5 reconstruction at cell interface (left-biased, i+1/2-)
 *
 * Given values v[i-2], v[i-1], v[i], v[i+1], v[i+2], computes the
 * reconstructed value at the right interface of cell i.
 *
 * @param v Array of 5 stencil values centered at i
 * @return Reconstructed value at i+1/2 (left state)
 */
inline double weno5_left(const double* v) {
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

    // Candidate polynomials (reconstructed values at i+1/2)
    double p0 = (2.0 * v[0] - 7.0 * v[1] + 11.0 * v[2]) / 6.0;
    double p1 = (-v[1] + 5.0 * v[2] + 2.0 * v[3]) / 6.0;
    double p2 = (2.0 * v[2] + 5.0 * v[3] - v[4]) / 6.0;

    // Weighted combination
    return omega0 * p0 + omega1 * p1 + omega2 * p2;
}

/**
 * @brief Compute WENO-5 reconstruction at cell interface (right-biased, i-1/2+)
 *
 * Given values v[i-2], v[i-1], v[i], v[i+1], v[i+2], computes the
 * reconstructed value at the left interface of cell i.
 *
 * @param v Array of 5 stencil values centered at i
 * @return Reconstructed value at i-1/2 (right state)
 */
inline double weno5_right(const double* v) {
    // Mirror the stencil for right-biased reconstruction
    double v_mirror[5] = {v[4], v[3], v[2], v[1], v[0]};

    // Smoothness indicators (same formula, mirrored data)
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

/**
 * @brief Compute upwind derivative using WENO-5 in x-direction
 *
 * @param G Level-set field array
 * @param i x-index (including ghost cells)
 * @param j y-index (including ghost cells)
 * @param u_eff Effective velocity in x-direction
 * @param dx Grid spacing in x
 * @param nx_total Total grid size in x
 * @return Upwind derivative dG/dx
 */
inline double weno5_dx(const double* G, int i, int j,
                       double u_eff, double dx, int nx_total) {
    // Gather stencil values
    double v[5];
    v[0] = G[idx(i - 2, j, nx_total)];
    v[1] = G[idx(i - 1, j, nx_total)];
    v[2] = G[idx(i, j, nx_total)];
    v[3] = G[idx(i + 1, j, nx_total)];
    v[4] = G[idx(i + 2, j, nx_total)];

    if (u_eff >= 0.0) {
        // Upwind from left: use left-biased reconstruction
        // dG/dx ≈ (G_{i+1/2}^- - G_{i-1/2}^-) / dx
        double v_left[5] = {
            G[idx(i - 3, j, nx_total)],
            G[idx(i - 2, j, nx_total)],
            G[idx(i - 1, j, nx_total)],
            G[idx(i, j, nx_total)],
            G[idx(i + 1, j, nx_total)]
        };
        double G_left = weno5_left(v_left);    // G at i-1/2
        double G_right = weno5_left(v);        // G at i+1/2
        return (G_right - G_left) / dx;
    } else {
        // Upwind from right: use right-biased reconstruction
        double v_right[5] = {
            G[idx(i - 1, j, nx_total)],
            G[idx(i, j, nx_total)],
            G[idx(i + 1, j, nx_total)],
            G[idx(i + 2, j, nx_total)],
            G[idx(i + 3, j, nx_total)]
        };
        double G_left = weno5_right(v);         // G at i-1/2
        double G_right = weno5_right(v_right);  // G at i+1/2
        return (G_right - G_left) / dx;
    }
}

/**
 * @brief Compute upwind derivative using WENO-5 in y-direction
 *
 * @param G Level-set field array
 * @param i x-index (including ghost cells)
 * @param j y-index (including ghost cells)
 * @param v_eff Effective velocity in y-direction
 * @param dy Grid spacing in y
 * @param nx_total Total grid size in x (for indexing)
 * @return Upwind derivative dG/dy
 */
inline double weno5_dy(const double* G, int i, int j,
                       double v_eff, double dy, int nx_total) {
    // Gather stencil values
    double v[5];
    v[0] = G[idx(i, j - 2, nx_total)];
    v[1] = G[idx(i, j - 1, nx_total)];
    v[2] = G[idx(i, j, nx_total)];
    v[3] = G[idx(i, j + 1, nx_total)];
    v[4] = G[idx(i, j + 2, nx_total)];

    if (v_eff >= 0.0) {
        // Upwind from bottom
        double v_bottom[5] = {
            G[idx(i, j - 3, nx_total)],
            G[idx(i, j - 2, nx_total)],
            G[idx(i, j - 1, nx_total)],
            G[idx(i, j, nx_total)],
            G[idx(i, j + 1, nx_total)]
        };
        double G_bottom = weno5_left(v_bottom);  // G at j-1/2
        double G_top = weno5_left(v);            // G at j+1/2
        return (G_top - G_bottom) / dy;
    } else {
        // Upwind from top
        double v_top[5] = {
            G[idx(i, j - 1, nx_total)],
            G[idx(i, j, nx_total)],
            G[idx(i, j + 1, nx_total)],
            G[idx(i, j + 2, nx_total)],
            G[idx(i, j + 3, nx_total)]
        };
        double G_bottom = weno5_right(v);        // G at j-1/2
        double G_top = weno5_right(v_top);       // G at j+1/2
        return (G_top - G_bottom) / dy;
    }
}

/**
 * @brief Compute gradient magnitude using WENO-5 (for reinitialization)
 *
 * Uses Godunov's upwind scheme for Hamilton-Jacobi equations
 *
 * @param G Level-set field array
 * @param i x-index
 * @param j y-index
 * @param sign Sign of the level-set function (for upwind direction)
 * @param dx, dy Grid spacings
 * @param nx_total Total grid size in x
 * @return Gradient magnitude |∇G|
 */
inline double weno5_gradient_magnitude(const double* G, int i, int j,
                                        double sign, double dx, double dy,
                                        int nx_total) {
    // Compute one-sided derivatives in x-direction
    double dG_dx_minus = (G[idx(i, j, nx_total)] - G[idx(i - 1, j, nx_total)]) / dx;
    double dG_dx_plus = (G[idx(i + 1, j, nx_total)] - G[idx(i, j, nx_total)]) / dx;

    // Compute one-sided derivatives in y-direction
    double dG_dy_minus = (G[idx(i, j, nx_total)] - G[idx(i, j - 1, nx_total)]) / dy;
    double dG_dy_plus = (G[idx(i, j + 1, nx_total)] - G[idx(i, j, nx_total)]) / dy;

    // Godunov scheme for Hamilton-Jacobi equations
    double grad_mag_sq;

    if (sign > 0.0) {
        // Expanding interface (phi > 0): use min of |grad|
        double ax = fmax(dG_dx_minus, 0.0);
        double bx = fmin(dG_dx_plus, 0.0);
        double ay = fmax(dG_dy_minus, 0.0);
        double by = fmin(dG_dy_plus, 0.0);

        grad_mag_sq = fmax(ax * ax, bx * bx) + fmax(ay * ay, by * by);
    } else {
        // Contracting interface (phi < 0): use max of |grad|
        double ax = fmin(dG_dx_minus, 0.0);
        double bx = fmax(dG_dx_plus, 0.0);
        double ay = fmin(dG_dy_minus, 0.0);
        double by = fmax(dG_dy_plus, 0.0);

        grad_mag_sq = fmax(ax * ax, bx * bx) + fmax(ay * ay, by * by);
    }

    return sqrt(grad_mag_sq);
}

/**
 * @brief Compute WENO-5 derivatives for gradient calculation (central differences fallback)
 */
inline void weno5_gradient(const double* G, int i, int j,
                           double dx, double dy, int nx_total,
                           double& dGdx, double& dGdy) {
    // Central differences using WENO reconstruction
    double v_x[5], v_y[5];

    // x-derivative
    v_x[0] = G[idx(i - 2, j, nx_total)];
    v_x[1] = G[idx(i - 1, j, nx_total)];
    v_x[2] = G[idx(i, j, nx_total)];
    v_x[3] = G[idx(i + 1, j, nx_total)];
    v_x[4] = G[idx(i + 2, j, nx_total)];

    double G_xp = weno5_left(v_x);   // G at i+1/2

    double v_xm[5] = {
        G[idx(i - 3, j, nx_total)],
        v_x[0], v_x[1], v_x[2], v_x[3]
    };
    double G_xm = weno5_left(v_xm);  // G at i-1/2

    dGdx = (G_xp - G_xm) / dx;

    // y-derivative
    v_y[0] = G[idx(i, j - 2, nx_total)];
    v_y[1] = G[idx(i, j - 1, nx_total)];
    v_y[2] = G[idx(i, j, nx_total)];
    v_y[3] = G[idx(i, j + 1, nx_total)];
    v_y[4] = G[idx(i, j + 2, nx_total)];

    double G_yp = weno5_left(v_y);   // G at j+1/2

    double v_ym[5] = {
        G[idx(i, j - 3, nx_total)],
        v_y[0], v_y[1], v_y[2], v_y[3]
    };
    double G_ym = weno5_left(v_ym);  // G at j-1/2

    dGdy = (G_yp - G_ym) / dy;
}

#endif // WENO5_H
