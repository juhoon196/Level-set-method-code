/**
 * @file weno5.h
 * @brief 5th-order WENO scheme implementation for 3D
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

constexpr double WENO_D0 = 0.1;
constexpr double WENO_D1 = 0.6;
constexpr double WENO_D2 = 0.3;

constexpr double WENO_EPSILON = 1.0e-6;

//=============================================================================
// WENO-5 Functions
//=============================================================================

inline void computeSmoothnessIndicators(const double* v, double* beta) {
    double d2_0 = v[2] - 2.0 * v[1] + v[0];
    beta[0] = (13.0 / 12.0) * d2_0 * d2_0 + 0.25 * (v[2] - v[0]) * (v[2] - v[0]);

    double d2_1 = v[3] - 2.0 * v[2] + v[1];
    beta[1] = (13.0 / 12.0) * d2_1 * d2_1 + 0.25 * (v[3] - v[1]) * (v[3] - v[1]);

    double d2_2 = v[4] - 2.0 * v[3] + v[2];
    beta[2] = (13.0 / 12.0) * d2_2 * d2_2 + 0.25 * (v[4] - v[2]) * (v[4] - v[2]);
}

inline double weno5_left(const double* v) {
    double beta[3];
    computeSmoothnessIndicators(v, beta);

    double alpha0 = 0.1 / ((WENO_EPSILON + beta[0]) * (WENO_EPSILON + beta[0]));
    double alpha1 = 0.6 / ((WENO_EPSILON + beta[1]) * (WENO_EPSILON + beta[1]));
    double alpha2 = 0.3 / ((WENO_EPSILON + beta[2]) * (WENO_EPSILON + beta[2]));

    double sum_alpha = alpha0 + alpha1 + alpha2;

    double omega0 = alpha0 / sum_alpha;
    double omega1 = alpha1 / sum_alpha;
    double omega2 = alpha2 / sum_alpha;

    double p0 = (2.0 * v[0] - 7.0 * v[1] + 11.0 * v[2]) / 6.0;
    double p1 = (-v[1] + 5.0 * v[2] + 2.0 * v[3]) / 6.0;
    double p2 = (2.0 * v[2] + 5.0 * v[3] - v[4]) / 6.0;

    return omega0 * p0 + omega1 * p1 + omega2 * p2;
}

inline double weno5_right(const double* v) {
    double v_mirror[5] = {v[4], v[3], v[2], v[1], v[0]};

    double beta[3];
    computeSmoothnessIndicators(v_mirror, beta);

    double alpha0 = 0.3 / ((WENO_EPSILON + beta[0]) * (WENO_EPSILON + beta[0]));
    double alpha1 = 0.6 / ((WENO_EPSILON + beta[1]) * (WENO_EPSILON + beta[1]));
    double alpha2 = 0.1 / ((WENO_EPSILON + beta[2]) * (WENO_EPSILON + beta[2]));

    double sum_alpha = alpha0 + alpha1 + alpha2;

    double omega0 = alpha0 / sum_alpha;
    double omega1 = alpha1 / sum_alpha;
    double omega2 = alpha2 / sum_alpha;

    double p0 = (11.0 * v[2] - 7.0 * v[3] + 2.0 * v[4]) / 6.0;
    double p1 = (2.0 * v[1] + 5.0 * v[2] - v[3]) / 6.0;
    double p2 = (-v[0] + 5.0 * v[1] + 2.0 * v[2]) / 6.0;

    return omega0 * p0 + omega1 * p1 + omega2 * p2;
}

//=============================================================================
// 3D WENO-5 Upwind Derivatives
//=============================================================================

/**
 * @brief Compute upwind derivative using WENO-5 in x-direction (3D)
 */
inline double weno5_dx_3d(const double* G, int i, int j, int k,
                           double u_eff, double dx, int nx_total, int ny_total) {
    double v[5];
    v[0] = G[idx3(i - 2, j, k, nx_total, ny_total)];
    v[1] = G[idx3(i - 1, j, k, nx_total, ny_total)];
    v[2] = G[idx3(i,     j, k, nx_total, ny_total)];
    v[3] = G[idx3(i + 1, j, k, nx_total, ny_total)];
    v[4] = G[idx3(i + 2, j, k, nx_total, ny_total)];

    if (u_eff >= 0.0) {
        double v_left[5] = {
            G[idx3(i - 3, j, k, nx_total, ny_total)],
            G[idx3(i - 2, j, k, nx_total, ny_total)],
            G[idx3(i - 1, j, k, nx_total, ny_total)],
            G[idx3(i,     j, k, nx_total, ny_total)],
            G[idx3(i + 1, j, k, nx_total, ny_total)]
        };
        double G_left = weno5_left(v_left);
        double G_right = weno5_left(v);
        return (G_right - G_left) / dx;
    } else {
        double v_right[5] = {
            G[idx3(i - 1, j, k, nx_total, ny_total)],
            G[idx3(i,     j, k, nx_total, ny_total)],
            G[idx3(i + 1, j, k, nx_total, ny_total)],
            G[idx3(i + 2, j, k, nx_total, ny_total)],
            G[idx3(i + 3, j, k, nx_total, ny_total)]
        };
        double G_left = weno5_right(v);
        double G_right = weno5_right(v_right);
        return (G_right - G_left) / dx;
    }
}

/**
 * @brief Compute upwind derivative using WENO-5 in y-direction (3D)
 */
inline double weno5_dy_3d(const double* G, int i, int j, int k,
                           double v_eff, double dy, int nx_total, int ny_total) {
    double v[5];
    v[0] = G[idx3(i, j - 2, k, nx_total, ny_total)];
    v[1] = G[idx3(i, j - 1, k, nx_total, ny_total)];
    v[2] = G[idx3(i, j,     k, nx_total, ny_total)];
    v[3] = G[idx3(i, j + 1, k, nx_total, ny_total)];
    v[4] = G[idx3(i, j + 2, k, nx_total, ny_total)];

    if (v_eff >= 0.0) {
        double v_bottom[5] = {
            G[idx3(i, j - 3, k, nx_total, ny_total)],
            G[idx3(i, j - 2, k, nx_total, ny_total)],
            G[idx3(i, j - 1, k, nx_total, ny_total)],
            G[idx3(i, j,     k, nx_total, ny_total)],
            G[idx3(i, j + 1, k, nx_total, ny_total)]
        };
        double G_bottom = weno5_left(v_bottom);
        double G_top = weno5_left(v);
        return (G_top - G_bottom) / dy;
    } else {
        double v_top[5] = {
            G[idx3(i, j - 1, k, nx_total, ny_total)],
            G[idx3(i, j,     k, nx_total, ny_total)],
            G[idx3(i, j + 1, k, nx_total, ny_total)],
            G[idx3(i, j + 2, k, nx_total, ny_total)],
            G[idx3(i, j + 3, k, nx_total, ny_total)]
        };
        double G_bottom = weno5_right(v);
        double G_top = weno5_right(v_top);
        return (G_top - G_bottom) / dy;
    }
}

/**
 * @brief Compute upwind derivative using WENO-5 in z-direction (3D)
 */
inline double weno5_dz_3d(const double* G, int i, int j, int k,
                           double w_eff, double dz, int nx_total, int ny_total) {
    double v[5];
    v[0] = G[idx3(i, j, k - 2, nx_total, ny_total)];
    v[1] = G[idx3(i, j, k - 1, nx_total, ny_total)];
    v[2] = G[idx3(i, j, k,     nx_total, ny_total)];
    v[3] = G[idx3(i, j, k + 1, nx_total, ny_total)];
    v[4] = G[idx3(i, j, k + 2, nx_total, ny_total)];

    if (w_eff >= 0.0) {
        double v_back[5] = {
            G[idx3(i, j, k - 3, nx_total, ny_total)],
            G[idx3(i, j, k - 2, nx_total, ny_total)],
            G[idx3(i, j, k - 1, nx_total, ny_total)],
            G[idx3(i, j, k,     nx_total, ny_total)],
            G[idx3(i, j, k + 1, nx_total, ny_total)]
        };
        double G_back = weno5_left(v_back);
        double G_front = weno5_left(v);
        return (G_front - G_back) / dz;
    } else {
        double v_front[5] = {
            G[idx3(i, j, k - 1, nx_total, ny_total)],
            G[idx3(i, j, k,     nx_total, ny_total)],
            G[idx3(i, j, k + 1, nx_total, ny_total)],
            G[idx3(i, j, k + 2, nx_total, ny_total)],
            G[idx3(i, j, k + 3, nx_total, ny_total)]
        };
        double G_back = weno5_right(v);
        double G_front = weno5_right(v_front);
        return (G_front - G_back) / dz;
    }
}

//=============================================================================
// 3D Gradient Magnitude (Godunov scheme for reinitialization)
//=============================================================================

inline double weno5_gradient_magnitude_3d(const double* G, int i, int j, int k,
                                           double sign, double dx, double dy, double dz,
                                           int nx_total, int ny_total) {
    // One-sided derivatives in x
    double dG_dx_minus = (G[idx3(i, j, k, nx_total, ny_total)] -
                          G[idx3(i - 1, j, k, nx_total, ny_total)]) / dx;
    double dG_dx_plus  = (G[idx3(i + 1, j, k, nx_total, ny_total)] -
                          G[idx3(i, j, k, nx_total, ny_total)]) / dx;

    // One-sided derivatives in y
    double dG_dy_minus = (G[idx3(i, j, k, nx_total, ny_total)] -
                          G[idx3(i, j - 1, k, nx_total, ny_total)]) / dy;
    double dG_dy_plus  = (G[idx3(i, j + 1, k, nx_total, ny_total)] -
                          G[idx3(i, j, k, nx_total, ny_total)]) / dy;

    // One-sided derivatives in z
    double dG_dz_minus = (G[idx3(i, j, k, nx_total, ny_total)] -
                          G[idx3(i, j, k - 1, nx_total, ny_total)]) / dz;
    double dG_dz_plus  = (G[idx3(i, j, k + 1, nx_total, ny_total)] -
                          G[idx3(i, j, k, nx_total, ny_total)]) / dz;

    double grad_mag_sq;

    if (sign > 0.0) {
        double ax = fmax(dG_dx_minus, 0.0);
        double bx = fmin(dG_dx_plus, 0.0);
        double ay = fmax(dG_dy_minus, 0.0);
        double by = fmin(dG_dy_plus, 0.0);
        double az = fmax(dG_dz_minus, 0.0);
        double bz = fmin(dG_dz_plus, 0.0);

        grad_mag_sq = fmax(ax * ax, bx * bx) + fmax(ay * ay, by * by)
                    + fmax(az * az, bz * bz);
    } else {
        double ax = fmin(dG_dx_minus, 0.0);
        double bx = fmax(dG_dx_plus, 0.0);
        double ay = fmin(dG_dy_minus, 0.0);
        double by = fmax(dG_dy_plus, 0.0);
        double az = fmin(dG_dz_minus, 0.0);
        double bz = fmax(dG_dz_plus, 0.0);

        grad_mag_sq = fmax(ax * ax, bx * bx) + fmax(ay * ay, by * by)
                    + fmax(az * az, bz * bz);
    }

    return sqrt(grad_mag_sq);
}

#endif // WENO5_H
