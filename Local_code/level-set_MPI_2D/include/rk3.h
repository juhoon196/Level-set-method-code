/**
 * @file rk3.h
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

#ifndef RK3_H
#define RK3_H

#include <cmath>
#include "config.h"
#include "weno5.h"
#include "boundary.h"

//=============================================================================
// RK3 Constants (Shu-Osher form)
//=============================================================================

// Stage coefficients
constexpr double RK3_ALPHA2_0 = 0.75;   // Coefficient for G^n in stage 2
constexpr double RK3_ALPHA2_1 = 0.25;   // Coefficient for G^(1) in stage 2
constexpr double RK3_BETA2 = 0.25;      // Coefficient for dt*L in stage 2

constexpr double RK3_ALPHA3_0 = 1.0 / 3.0;  // Coefficient for G^n in stage 3
constexpr double RK3_ALPHA3_2 = 2.0 / 3.0;  // Coefficient for G^(2) in stage 3
constexpr double RK3_BETA3 = 2.0 / 3.0;     // Coefficient for dt*L in stage 3

//=============================================================================
// RHS (Spatial Operator) Computation
//=============================================================================

/**
 * @brief Compute the right-hand side of the G-equation: -u_eff · ∇G
 *
 * The G-equation is: ∂G/∂t + u_eff · ∇G = 0
 * where u_eff = u - S_L * (∇G / |∇G|)
 *
 * @param G Level-set field (local array)
 * @param G_rhs Output RHS array
 * @param u Velocity field (x-component)
 * @param v Velocity field (y-component)
 * @param params Simulation parameters
 */
inline void computeRHS(const double* G, double* G_rhs,
                       const double* u, const double* v,
                       SimParams& params) {
    int nghost = params.nghost;
    int nx = params.nx;
    int local_ny = params.local_ny;
    int nx_total = params.nx_total;

    // Loop over interior points only
    for (int j = nghost; j < local_ny + nghost; j++) {
        for (int i = nghost; i < nx + nghost; i++) {
            int index = idx(i, j, nx_total);

            // Get local velocity
            double u_local = u[index];
            double v_local = v[index];

            // Compute gradient for flame speed term if S_L > 0
            double dGdx_central = 0.0, dGdy_central = 0.0;
            double grad_mag = 1.0;  // Default to avoid division by zero

            if (params.s_l > params.epsilon) {
                // Central difference for gradient direction
                dGdx_central = (G[idx(i + 1, j, nx_total)] -
                                G[idx(i - 1, j, nx_total)]) / (2.0 * params.dx);
                dGdy_central = (G[idx(i, j + 1, nx_total)] -
                                G[idx(i, j - 1, nx_total)]) / (2.0 * params.dy);

                grad_mag = sqrt(dGdx_central * dGdx_central +
                                dGdy_central * dGdy_central + params.epsilon);
            }

            // Compute effective velocity: u_eff = u - S_L * (∇G / |∇G|)
            double u_eff = u_local - params.s_l * dGdx_central / grad_mag;
            double v_eff = v_local - params.s_l * dGdy_central / grad_mag;

            // Compute upwind derivatives using WENO-5
            double dGdx = weno5_dx(G, i, j, u_eff, params.dx, nx_total);
            double dGdy = weno5_dy(G, i, j, v_eff, params.dy, nx_total);

            // RHS = -u_eff · ∇G
            G_rhs[index] = -(u_eff * dGdx + v_eff * dGdy);
        }
    }
}

/**
 * @brief RK3 Stage 1: G^(1) = G^n + dt * L(G^n)
 */
inline void rk3Stage1(const double* G_n, const double* G_rhs, double* G_1,
                      double dt, SimParams& params) {
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    for (int j = 0; j < local_ny_total; j++) {
        for (int i = 0; i < nx_total; i++) {
            int index = idx(i, j, nx_total);
            G_1[index] = G_n[index] + dt * G_rhs[index];
        }
    }
}

/**
 * @brief RK3 Stage 2: G^(2) = (3/4)*G^n + (1/4)*G^(1) + (1/4)*dt*L(G^(1))
 */
inline void rk3Stage2(const double* G_n, const double* G_1, const double* G_rhs,
                      double* G_2, double dt, SimParams& params) {
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    for (int j = 0; j < local_ny_total; j++) {
        for (int i = 0; i < nx_total; i++) {
            int index = idx(i, j, nx_total);
            G_2[index] = 0.75 * G_n[index] + 0.25 * G_1[index] + 0.25 * dt * G_rhs[index];
        }
    }
}

/**
 * @brief RK3 Stage 3: G^(n+1) = (1/3)*G^n + (2/3)*G^(2) + (2/3)*dt*L(G^(2))
 */
inline void rk3Stage3(const double* G_n, const double* G_2, const double* G_rhs,
                      double* G_new, double dt, SimParams& params) {
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    double alpha0 = 1.0 / 3.0;
    double alpha2 = 2.0 / 3.0;
    double beta3 = 2.0 / 3.0;

    for (int j = 0; j < local_ny_total; j++) {
        for (int i = 0; i < nx_total; i++) {
            int index = idx(i, j, nx_total);
            G_new[index] = alpha0 * G_n[index] + alpha2 * G_2[index] + beta3 * dt * G_rhs[index];
        }
    }
}

//=============================================================================
// Full RK3 Time Step (Host Function)
//=============================================================================

/**
 * @brief Perform one complete RK3 time step
 *
 * @param G_n Current solution
 * @param G_new New solution (output)
 * @param G_1 Temporary storage for stage 1
 * @param G_2 Temporary storage for stage 2
 * @param G_rhs Temporary storage for RHS
 * @param u Velocity field x-component
 * @param v Velocity field y-component
 * @param params Simulation parameters
 */
inline void rk3TimeStep(double* G_n, double* G_new,
                        double* G_1, double* G_2, double* G_rhs,
                        const double* u, const double* v,
                        SimParams& params) {

    // Stage 1: G^(1) = G^n + dt * L(G^n)
    computeRHS(G_n, G_rhs, u, v, params);
    rk3Stage1(G_n, G_rhs, G_1, params.dt, params);
    applyBoundaryConditions(G_1, params);

    // Stage 2: G^(2) = (3/4)*G^n + (1/4)*G^(1) + (1/4)*dt*L(G^(1))
    computeRHS(G_1, G_rhs, u, v, params);
    rk3Stage2(G_n, G_1, G_rhs, G_2, params.dt, params);
    applyBoundaryConditions(G_2, params);

    // Stage 3: G^(n+1) = (1/3)*G^n + (2/3)*G^(2) + (2/3)*dt*L(G^(2))
    computeRHS(G_2, G_rhs, u, v, params);
    rk3Stage3(G_n, G_2, G_rhs, G_new, params.dt, params);
    applyBoundaryConditions(G_new, params);
}

#endif // RK3_H
