/**
 * @file reinitialization.h
 * @brief Hartmann HCR-2 reinitialization for 3D level-set
 *
 * Solves (Eq. 15):
 *   ∂φ/∂τ + S(φ₀)(|∇φ| - 1) = β * F
 *
 * Extended to 3D with 6 neighbors (±x, ±y, ±z).
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

inline double smoothSign(double phi, double dx) {
    return phi / sqrt(phi * phi + dx * dx);
}

//=============================================================================
// Interface Detection
//=============================================================================

inline bool interfaceCrosses(double phi1, double phi2) {
    return phi1 * phi2 < 0.0;
}

//=============================================================================
// HCR-2 Pre-computation (3D)
//=============================================================================

/**
 * @brief Pre-compute r̃ for HCR-2 constraint (3D)
 *
 * interface_flag bits:
 *   1: x-minus, 2: x-plus, 4: y-minus, 8: y-plus, 16: z-minus, 32: z-plus
 */
inline void computeInterfaceCrossings3D(const double* phi_0,
                                         double* r_tilde,
                                         int* interface_flag,
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

                double phi_c  = phi_0[index];
                double phi_xm = phi_0[idx3(i - 1, j, k, nx_total, local_ny_total)];
                double phi_xp = phi_0[idx3(i + 1, j, k, nx_total, local_ny_total)];
                double phi_ym = phi_0[idx3(i, j - 1, k, nx_total, local_ny_total)];
                double phi_yp = phi_0[idx3(i, j + 1, k, nx_total, local_ny_total)];
                double phi_zm = phi_0[idx3(i, j, k - 1, nx_total, local_ny_total)];
                double phi_zp = phi_0[idx3(i, j, k + 1, nx_total, local_ny_total)];

                r_tilde[index] = 0.0;
                interface_flag[index] = 0;

                double sum_phi = 0.0;
                int count = 0;

                if (interfaceCrosses(phi_c, phi_xm)) { sum_phi += phi_xm; count++; interface_flag[index] |= 1; }
                if (interfaceCrosses(phi_c, phi_xp)) { sum_phi += phi_xp; count++; interface_flag[index] |= 2; }
                if (interfaceCrosses(phi_c, phi_ym)) { sum_phi += phi_ym; count++; interface_flag[index] |= 4; }
                if (interfaceCrosses(phi_c, phi_yp)) { sum_phi += phi_yp; count++; interface_flag[index] |= 8; }
                if (interfaceCrosses(phi_c, phi_zm)) { sum_phi += phi_zm; count++; interface_flag[index] |= 16; }
                if (interfaceCrosses(phi_c, phi_zp)) { sum_phi += phi_zp; count++; interface_flag[index] |= 32; }

                if (count > 0 && fabs(sum_phi) > 1e-15) {
                    r_tilde[index] = phi_c / sum_phi;
                }
            }
        }
    }
}

//=============================================================================
// Stability Constraint Check (3D)
//=============================================================================

inline bool checkStabilityConstraint3D(const double* phi_n,
                                        const double* phi_0,
                                        int i, int j, int k,
                                        int flag,
                                        int nx_total, int ny_total) {
    int index = idx3(i, j, k, nx_total, ny_total);

    if ((phi_n[index] > 0.0) != (phi_0[index] > 0.0)) return false;

    if (flag & 1) {
        int ni = idx3(i - 1, j, k, nx_total, ny_total);
        if ((phi_n[ni] > 0.0) != (phi_0[ni] > 0.0)) return false;
    }
    if (flag & 2) {
        int ni = idx3(i + 1, j, k, nx_total, ny_total);
        if ((phi_n[ni] > 0.0) != (phi_0[ni] > 0.0)) return false;
    }
    if (flag & 4) {
        int ni = idx3(i, j - 1, k, nx_total, ny_total);
        if ((phi_n[ni] > 0.0) != (phi_0[ni] > 0.0)) return false;
    }
    if (flag & 8) {
        int ni = idx3(i, j + 1, k, nx_total, ny_total);
        if ((phi_n[ni] > 0.0) != (phi_0[ni] > 0.0)) return false;
    }
    if (flag & 16) {
        int ni = idx3(i, j, k - 1, nx_total, ny_total);
        if ((phi_n[ni] > 0.0) != (phi_0[ni] > 0.0)) return false;
    }
    if (flag & 32) {
        int ni = idx3(i, j, k + 1, nx_total, ny_total);
        if ((phi_n[ni] > 0.0) != (phi_0[ni] > 0.0)) return false;
    }

    return true;
}

//=============================================================================
// HCR-2 Reinitialization Step (3D)
//=============================================================================

inline void reinitStep3D(const double* phi_n,
                          double* phi_new,
                          const double* phi_0,
                          const double* r_tilde,
                          const int* interface_flag,
                          double dtau,
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

                double phi = phi_n[index];
                double phi0 = phi_0[index];

                double S = smoothSign(phi0, params.dx);

                double grad_mag = weno5_gradient_magnitude_3d(
                    phi_n, i, j, k, S,
                    params.dx, params.dy, params.dz,
                    nx_total, local_ny_total);

                double reinit_term = -S * (grad_mag - 1.0);

                // HCR-2 forcing
                double forcing = 0.0;
                int flag = interface_flag[index];

                if (flag != 0) {
                    bool stable = checkStabilityConstraint3D(
                        phi_n, phi_0, i, j, k, flag, nx_total, local_ny_total);

                    if (stable) {
                        double sum_phi = 0.0;
                        if (flag & 1)  sum_phi += phi_n[idx3(i - 1, j, k, nx_total, local_ny_total)];
                        if (flag & 2)  sum_phi += phi_n[idx3(i + 1, j, k, nx_total, local_ny_total)];
                        if (flag & 4)  sum_phi += phi_n[idx3(i, j - 1, k, nx_total, local_ny_total)];
                        if (flag & 8)  sum_phi += phi_n[idx3(i, j + 1, k, nx_total, local_ny_total)];
                        if (flag & 16) sum_phi += phi_n[idx3(i, j, k - 1, nx_total, local_ny_total)];
                        if (flag & 32) sum_phi += phi_n[idx3(i, j, k + 1, nx_total, local_ny_total)];

                        double r = r_tilde[index];
                        forcing = params.reinit_beta * (r * sum_phi - phi) / params.dx;
                    }
                }

                phi_new[index] = phi + dtau * (reinit_term + forcing);
            }
        }
    }
}

//=============================================================================
// Full Reinitialization (3D)
//=============================================================================

inline void reinitializeWithSwap3D(double** phi_ptr, double** phi_temp_ptr, double* phi_0,
                                    double* r_tilde, int* interface_flag,
                                    SimParams& params) {
    if (!params.enable_reinit) return;

    int total_local_size = params.nx_total * params.local_ny_total * params.nz_total;

    double* phi = *phi_ptr;
    double* phi_temp = *phi_temp_ptr;

    memcpy(phi_0, phi, total_local_size * sizeof(double));
    applyBoundaryConditions3D(phi_0, params);

    computeInterfaceCrossings3D(phi_0, r_tilde, interface_flag, params);

    double dtau = params.reinit_dtau;

    for (int iter = 0; iter < params.reinit_iterations; iter++) {
        reinitStep3D(phi, phi_temp, phi_0, r_tilde, interface_flag, dtau, params);

        double* temp = phi;
        phi = phi_temp;
        phi_temp = temp;

        applyBoundaryConditions3D(phi, params);
    }

    *phi_ptr = phi;
    *phi_temp_ptr = phi_temp;
}

#endif // REINITIALIZATION_H
