/**
 * @file reinitialization.h
 * @brief Hartmann HCR-2 (High-order Constrained Reinitialization) implementation
 *
 * This module implements the HCR-2 reinitialization method from:
 * Hartmann et al. (2010), "The constrained reinitialization equation for
 * level set methods"
 *
 * The method solves (Eq. 15):
 *   ∂φ/∂τ + S(φ₀)(|∇φ| - 1) = β * F
 *
 * where F is the HCR-2 forcing term (Eq. 21b):
 *   F_{i,j} = (1/Δx) * (r̃_{i,j} * Σφ_neighbors - φ_{i,j})
 *
 * and r̃ is pre-computed from initial values (Eq. 19b):
 *   r̃_{i,j} = φ₀_{i,j} / Σφ₀_neighbors
 */

#ifndef REINITIALIZATION_H
#define REINITIALIZATION_H

#include <cmath>
#include <cstring>
#include "config.h"
#include "weno5.h"
#include "boundary.h"

//=============================================================================
// Sign Function
//=============================================================================

/**
 * @brief Smoothed sign function for reinitialization
 *
 * S(φ) = φ / sqrt(φ² + Δx²)
 */
inline double smoothSign(double phi, double dx) {
    return phi / sqrt(phi * phi + dx * dx);
}

//=============================================================================
// Interface Detection
//=============================================================================

/**
 * @brief Check if cell (i,j) is near the interface
 */
inline bool isNearInterface(const double* phi, int i, int j,
                            int nx_total, double band_width, double dx) {
    return fabs(phi[idx(i, j, nx_total)]) < band_width * dx;
}

/**
 * @brief Check if interface crosses between two cells (different signs)
 */
inline bool interfaceCrosses(double phi1, double phi2) {
    return phi1 * phi2 < 0.0;
}

//=============================================================================
// HCR-2 Pre-computation (Eq. 19b)
//=============================================================================

/**
 * @brief Pre-compute r̃ for HCR-2 constraint
 *
 * According to Eq. 19b:
 *   r̃_{i,j} = φ₀_{i,j} / Σ_{m ∈ N_{i,j}} φ₀_m
 *
 * @param phi_0 Initial level-set field before reinitialization
 * @param r_tilde Pre-computed r̃ values (output)
 * @param interface_flag Flag indicating interface-adjacent cells (output)
 * @param params Simulation parameters
 */
inline void computeInterfaceCrossings(const double* phi_0,
                                      double* r_tilde,
                                      int* interface_flag,
                                      SimParams& params) {
    int nghost = params.nghost;
    int nx = params.nx;
    int local_ny = params.local_ny;
    int nx_total = params.nx_total;

    for (int j = nghost; j < local_ny + nghost; j++) {
        for (int i = nghost; i < nx + nghost; i++) {
            int index = idx(i, j, nx_total);

            // Get current and neighboring values
            double phi_c = phi_0[index];
            double phi_xm = phi_0[idx(i - 1, j, nx_total)];
            double phi_xp = phi_0[idx(i + 1, j, nx_total)];
            double phi_ym = phi_0[idx(i, j - 1, nx_total)];
            double phi_yp = phi_0[idx(i, j + 1, nx_total)];

            // Initialize
            r_tilde[index] = 0.0;
            interface_flag[index] = 0;

            // Sum neighbors with interface crossings (opposite sign from phi_c)
            double sum_phi_neighbors = 0.0;
            int neighbor_count = 0;

            if (interfaceCrosses(phi_c, phi_xm)) {
                sum_phi_neighbors += phi_xm;
                neighbor_count++;
                interface_flag[index] |= 1;  // x-minus crossing
            }
            if (interfaceCrosses(phi_c, phi_xp)) {
                sum_phi_neighbors += phi_xp;
                neighbor_count++;
                interface_flag[index] |= 2;  // x-plus crossing
            }
            if (interfaceCrosses(phi_c, phi_ym)) {
                sum_phi_neighbors += phi_ym;
                neighbor_count++;
                interface_flag[index] |= 4;  // y-minus crossing
            }
            if (interfaceCrosses(phi_c, phi_yp)) {
                sum_phi_neighbors += phi_yp;
                neighbor_count++;
                interface_flag[index] |= 8;  // y-plus crossing
            }

            // Compute r̃ = φ₀ / Σφ₀_neighbors (Eq. 19b)
            if (neighbor_count > 0 && fabs(sum_phi_neighbors) > 1e-15) {
                r_tilde[index] = phi_c / sum_phi_neighbors;
            }
        }
    }
}

//=============================================================================
// Godunov Gradient with WENO-5
//=============================================================================

/**
 * @brief Compute gradient magnitude using Godunov's scheme with WENO-5 derivatives
 */
inline double godunovGradientWENO5(const double* phi, int i, int j,
                                   double sign_phi0, double dx, double dy,
                                   int nx_total) {
    return weno5_gradient_magnitude(phi, i, j, sign_phi0, dx, dy, nx_total);
}

//=============================================================================
// Stability Constraint Check (Eq. 18)
//=============================================================================

/**
 * @brief Check if the cell and its crossing neighbors satisfy the stability constraint C^v
 */
inline bool checkStabilityConstraint(const double* phi_n,
                                     const double* phi_0,
                                     int i, int j,
                                     int interface_flag,
                                     int nx_total) {
    int index = idx(i, j, nx_total);

    // Check current cell
    double phi_c = phi_n[index];
    double phi0_c = phi_0[index];
    if ((phi_c > 0.0) != (phi0_c > 0.0)) {
        return false;  // Current cell changed sign
    }

    // Check neighbors that cross the interface
    if (interface_flag & 1) {  // x-minus
        double phi_xm = phi_n[idx(i - 1, j, nx_total)];
        double phi0_xm = phi_0[idx(i - 1, j, nx_total)];
        if ((phi_xm > 0.0) != (phi0_xm > 0.0)) return false;
    }
    if (interface_flag & 2) {  // x-plus
        double phi_xp = phi_n[idx(i + 1, j, nx_total)];
        double phi0_xp = phi_0[idx(i + 1, j, nx_total)];
        if ((phi_xp > 0.0) != (phi0_xp > 0.0)) return false;
    }
    if (interface_flag & 4) {  // y-minus
        double phi_ym = phi_n[idx(i, j - 1, nx_total)];
        double phi0_ym = phi_0[idx(i, j - 1, nx_total)];
        if ((phi_ym > 0.0) != (phi0_ym > 0.0)) return false;
    }
    if (interface_flag & 8) {  // y-plus
        double phi_yp = phi_n[idx(i, j + 1, nx_total)];
        double phi0_yp = phi_0[idx(i, j + 1, nx_total)];
        if ((phi_yp > 0.0) != (phi0_yp > 0.0)) return false;
    }

    return true;
}

//=============================================================================
// HCR-2 Reinitialization Step
//=============================================================================

/**
 * @brief Single pseudo-time step of HCR-2 reinitialization
 *
 * Solves (Eq. 15): ∂φ/∂τ + S(φ₀)(|∇φ| - 1) = β * F
 */
inline void reinitStep(const double* phi_n,
                       double* phi_new,
                       const double* phi_0,
                       const double* r_tilde,
                       const int* interface_flag,
                       double dtau,
                       SimParams& params) {
    int nghost = params.nghost;
    int nx = params.nx;
    int local_ny = params.local_ny;
    int nx_total = params.nx_total;

    for (int j = nghost; j < local_ny + nghost; j++) {
        for (int i = nghost; i < nx + nghost; i++) {
            int index = idx(i, j, nx_total);

            double phi = phi_n[index];
            double phi0 = phi_0[index];

            // Smoothed sign function
            double S = smoothSign(phi0, params.dx);

            // Compute gradient magnitude using WENO-5 with Godunov Hamiltonian
            double grad_mag = godunovGradientWENO5(phi_n, i, j, S, params.dx, params.dy, nx_total);

            // Standard reinitialization term: -S(φ₀)(|∇φ| - 1)
            double reinit_term = -S * (grad_mag - 1.0);

            // HCR-2 forcing term (Eq. 21b)
            double forcing = 0.0;
            int flag = interface_flag[index];

            if (flag != 0) {
                // Check stability constraint C^v (Eq. 18)
                bool stable = checkStabilityConstraint(phi_n, phi_0, i, j, flag, nx_total);

                if (stable) {
                    // Compute sum of neighboring φ values (only those with interface crossings)
                    double sum_phi_neighbors = 0.0;

                    if (flag & 1) {  // x-minus
                        sum_phi_neighbors += phi_n[idx(i - 1, j, nx_total)];
                    }
                    if (flag & 2) {  // x-plus
                        sum_phi_neighbors += phi_n[idx(i + 1, j, nx_total)];
                    }
                    if (flag & 4) {  // y-minus
                        sum_phi_neighbors += phi_n[idx(i, j - 1, nx_total)];
                    }
                    if (flag & 8) {  // y-plus
                        sum_phi_neighbors += phi_n[idx(i, j + 1, nx_total)];
                    }

                    // HCR-2 forcing (Eq. 21b): F = (1/Δx) * (r̃ * Σφ_neighbors - φ)
                    double r = r_tilde[index];
                    forcing = params.reinit_beta * (r * sum_phi_neighbors - phi) / params.dx;
                }
            }

            // Update: φ^{n+1} = φ^n + Δτ * (reinit_term + forcing)
            phi_new[index] = phi + dtau * (reinit_term + forcing);
        }
    }
}

//=============================================================================
// Full Reinitialization Procedure
//=============================================================================

/**
 * @brief Perform complete HCR-2 reinitialization
 *
 * @param phi Current level-set field (will be updated in place via pointer swap)
 * @param phi_temp Temporary storage
 * @param phi_0 Copy of original field before reinitialization
 * @param r_tilde Pre-computed r̃ values
 * @param interface_flag Interface-adjacent cell flags
 * @param params Simulation parameters
 */
inline void reinitialize(double* phi, double* phi_temp, double* phi_0,
                         double* r_tilde, int* interface_flag,
                         SimParams& params) {

    if (!params.enable_reinit) return;

    int total_local_size = params.nx_total * params.local_ny_total;

    // Copy current field to phi_0 for reference
    memcpy(phi_0, phi, total_local_size * sizeof(double));

    // Apply boundary conditions to ensure ghost cells are valid
    applyBoundaryConditions(phi_0, params);

    // Pre-compute interface crossings and r̃ values
    computeInterfaceCrossings(phi_0, r_tilde, interface_flag, params);

    // Pseudo-time stepping
    double dtau = params.reinit_dtau;

    for (int iter = 0; iter < params.reinit_iterations; iter++) {
        // Perform one reinitialization step
        reinitStep(phi, phi_temp, phi_0, r_tilde, interface_flag, dtau, params);

        // Swap pointers
        double* temp = phi;
        phi = phi_temp;
        phi_temp = temp;

        // Apply boundary conditions
        applyBoundaryConditions(phi, params);
    }

    // If odd number of iterations, copy back to original
    if (params.reinit_iterations % 2 == 1) {
        memcpy(phi_temp, phi, total_local_size * sizeof(double));
    }
}

/**
 * @brief Alternative reinitialization interface that handles pointer swapping
 */
inline void reinitializeWithSwap(double** phi_ptr, double** phi_temp_ptr, double* phi_0,
                                 double* r_tilde, int* interface_flag,
                                 SimParams& params) {

    if (!params.enable_reinit) return;

    int total_local_size = params.nx_total * params.local_ny_total;

    double* phi = *phi_ptr;
    double* phi_temp = *phi_temp_ptr;

    // Copy current field to phi_0 for reference
    memcpy(phi_0, phi, total_local_size * sizeof(double));

    // Apply boundary conditions to ensure ghost cells are valid
    applyBoundaryConditions(phi_0, params);

    // Pre-compute interface crossings and r̃ values
    computeInterfaceCrossings(phi_0, r_tilde, interface_flag, params);

    // Pseudo-time stepping
    double dtau = params.reinit_dtau;

    for (int iter = 0; iter < params.reinit_iterations; iter++) {
        // Perform one reinitialization step
        reinitStep(phi, phi_temp, phi_0, r_tilde, interface_flag, dtau, params);

        // Swap pointers
        double* temp = phi;
        phi = phi_temp;
        phi_temp = temp;

        // Apply boundary conditions
        applyBoundaryConditions(phi, params);
    }

    // Update output pointers
    *phi_ptr = phi;
    *phi_temp_ptr = phi_temp;
}

#endif // REINITIALIZATION_H
