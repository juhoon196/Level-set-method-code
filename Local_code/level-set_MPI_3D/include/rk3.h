/**
 * @file rk3.h
 * @brief 3rd-order TVD Runge-Kutta time integration (3D version)
 *
 * The scheme is:
 *   Stage 1: G^(1) = G^n + dt * L(G^n)
 *   Stage 2: G^(2) = (3/4)*G^n + (1/4)*G^(1) + (1/4)*dt*L(G^(1))
 *   Stage 3: G^(n+1) = (1/3)*G^n + (2/3)*G^(2) + (2/3)*dt*L(G^(2))
 *
 * Reference: Shu & Osher (1988)
 */

#ifndef RK3_H
#define RK3_H

#include <cmath>
#include "config.h"
#include "weno5.h"
#include "boundary.h"

//=============================================================================
// RHS (Spatial Operator) Computation (3D)
//=============================================================================

/**
 * @brief Compute the right-hand side of the G-equation in 3D: -u_eff · ∇G
 *
 * G-equation: ∂G/∂t + u_eff · ∇G = 0
 * where u_eff = (u, v, w) - S_L * (∇G / |∇G|)
 */
inline void computeRHS3D(const double* G, double* G_rhs,
                          const double* u, const double* v, const double* w,
                          SimParams& params) {
    int nghost = params.nghost;
    int nx = params.nx;
    int nz = params.nz;
    int local_ny = params.local_ny;
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    for (int k = nghost; k < nz + nghost; k++) {
        for (int j = nghost; j < local_ny + nghost; j++) {
            for (int i = nghost; i < nx + nghost; i++) {
                int index = idx3(i, j, k, nx_total, local_ny_total);

                double u_local = u[index];
                double v_local = v[index];
                double w_local = w[index];

                // Gradient for flame speed term
                double dGdx_central = 0.0, dGdy_central = 0.0, dGdz_central = 0.0;
                double grad_mag = 1.0;

                if (params.s_l > params.epsilon) {
                    dGdx_central = (G[idx3(i + 1, j, k, nx_total, local_ny_total)] -
                                    G[idx3(i - 1, j, k, nx_total, local_ny_total)]) / (2.0 * params.dx);
                    dGdy_central = (G[idx3(i, j + 1, k, nx_total, local_ny_total)] -
                                    G[idx3(i, j - 1, k, nx_total, local_ny_total)]) / (2.0 * params.dy);
                    dGdz_central = (G[idx3(i, j, k + 1, nx_total, local_ny_total)] -
                                    G[idx3(i, j, k - 1, nx_total, local_ny_total)]) / (2.0 * params.dz);

                    grad_mag = sqrt(dGdx_central * dGdx_central +
                                    dGdy_central * dGdy_central +
                                    dGdz_central * dGdz_central + params.epsilon);
                }

                double u_eff = u_local - params.s_l * dGdx_central / grad_mag;
                double v_eff = v_local - params.s_l * dGdy_central / grad_mag;
                double w_eff = w_local - params.s_l * dGdz_central / grad_mag;

                // WENO-5 upwind derivatives
                double dGdx = weno5_dx_3d(G, i, j, k, u_eff, params.dx, nx_total, local_ny_total);
                double dGdy = weno5_dy_3d(G, i, j, k, v_eff, params.dy, nx_total, local_ny_total);
                double dGdz = weno5_dz_3d(G, i, j, k, w_eff, params.dz, nx_total, local_ny_total);

                // RHS = -u_eff · ∇G
                G_rhs[index] = -(u_eff * dGdx + v_eff * dGdy + w_eff * dGdz);
            }
        }
    }
}

//=============================================================================
// RK3 Stages (3D)
//=============================================================================

inline void rk3Stage1_3D(const double* G_n, const double* G_rhs, double* G_1,
                          double dt, SimParams& params) {
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;
    int nz_total = params.nz_total;
    int total = nx_total * local_ny_total * nz_total;

    for (int idx = 0; idx < total; idx++) {
        G_1[idx] = G_n[idx] + dt * G_rhs[idx];
    }
}

inline void rk3Stage2_3D(const double* G_n, const double* G_1, const double* G_rhs,
                          double* G_2, double dt, SimParams& params) {
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;
    int nz_total = params.nz_total;
    int total = nx_total * local_ny_total * nz_total;

    for (int idx = 0; idx < total; idx++) {
        G_2[idx] = 0.75 * G_n[idx] + 0.25 * G_1[idx] + 0.25 * dt * G_rhs[idx];
    }
}

inline void rk3Stage3_3D(const double* G_n, const double* G_2, const double* G_rhs,
                          double* G_new, double dt, SimParams& params) {
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;
    int nz_total = params.nz_total;
    int total = nx_total * local_ny_total * nz_total;

    double alpha0 = 1.0 / 3.0;
    double alpha2 = 2.0 / 3.0;
    double beta3 = 2.0 / 3.0;

    for (int idx = 0; idx < total; idx++) {
        G_new[idx] = alpha0 * G_n[idx] + alpha2 * G_2[idx] + beta3 * dt * G_rhs[idx];
    }
}

//=============================================================================
// Full RK3 Time Step (3D)
//=============================================================================

inline void rk3TimeStep3D(double* G_n, double* G_new,
                           double* G_1, double* G_2, double* G_rhs,
                           const double* u, const double* v, const double* w,
                           SimParams& params) {

    // Stage 1
    computeRHS3D(G_n, G_rhs, u, v, w, params);
    rk3Stage1_3D(G_n, G_rhs, G_1, params.dt, params);
    applyBoundaryConditions3D(G_1, params);

    // Stage 2
    computeRHS3D(G_1, G_rhs, u, v, w, params);
    rk3Stage2_3D(G_n, G_1, G_rhs, G_2, params.dt, params);
    applyBoundaryConditions3D(G_2, params);

    // Stage 3
    computeRHS3D(G_2, G_rhs, u, v, w, params);
    rk3Stage3_3D(G_n, G_2, G_rhs, G_new, params.dt, params);
    applyBoundaryConditions3D(G_new, params);
}

#endif // RK3_H
