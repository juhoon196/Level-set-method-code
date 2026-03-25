/**
 * @file reinitialization.cuh
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
 *
 * The forcing is only applied within the constraint subspace C^v (Eq. 18)
 * where all neighbors and the current cell have the same sign as φ₀.
 */

#ifndef REINITIALIZATION_CUH
#define REINITIALIZATION_CUH

#include <cuda_runtime.h>
#include "config.cuh"
#include "weno5.cuh"

//=============================================================================
// Sign Function
//=============================================================================

/**
 * @brief Smoothed sign function for reinitialization
 *
 * S(φ) = φ / sqrt(φ² + Δx²)
 *
 * This smoothed version avoids numerical issues at the interface
 */
__device__ inline double smoothSign(double phi, double dx) {
    return phi / sqrt(phi * phi + dx * dx);
}

//=============================================================================
// Interface Detection
//=============================================================================

/**
 * @brief Check if cell (i,j) is near the interface
 */
__device__ inline bool isNearInterface(const double* phi, int i, int j,
                                        int nx_total, double band_width, double dx) {
    return fabs(phi[idx(i, j, nx_total)]) < band_width * dx;
}

/**
 * @brief Check if interface crosses between two cells (different signs)
 */
__device__ inline bool interfaceCrosses(double phi1, double phi2) {
    return phi1 * phi2 < 0.0;
}

//=============================================================================
// HCR-2 Pre-computation Kernel (Eq. 19b)
//=============================================================================

/**
 * @brief Pre-compute r̃ for HCR-2 constraint
 *
 * According to Eq. 19b:
 *   r̃_{i,j} = φ₀_{i,j} / Σ_{m ∈ N_{i,j}} φ₀_m
 *
 * where N_{i,j} are neighbors with interface crossings (opposite sign).
 * This kernel stores r̃ and flags cells adjacent to the interface.
 *
 * @param phi_0 Initial level-set field before reinitialization
 * @param r_tilde Pre-computed r̃ values
 * @param interface_flag Flag indicating interface-adjacent cells
 * @param params Simulation parameters
 */
__global__ void computeInterfaceCrossings(const double* __restrict__ phi_0,
                                           double* __restrict__ r_tilde,
                                           int* __restrict__ interface_flag,
                                           SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process interior points
    if (i < params.nghost || i >= params.nx + params.nghost ||
        j < params.nghost || j >= params.ny + params.nghost) {
        return;
    }

    int index = idx(i, j, params.nx_total);

    // Get current and neighboring values
    double phi_c = phi_0[index];
    double phi_xm = phi_0[idx(i - 1, j, params.nx_total)];
    double phi_xp = phi_0[idx(i + 1, j, params.nx_total)];
    double phi_ym = phi_0[idx(i, j - 1, params.nx_total)];
    double phi_yp = phi_0[idx(i, j + 1, params.nx_total)];

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

//=============================================================================
// Godunov Gradient with WENO-5
//=============================================================================

/**
 * @brief Compute gradient magnitude using Godunov's scheme with WENO-5 derivatives
 *
 * For sign(φ₀) > 0: Use upwind from positive to negative
 * For sign(φ₀) < 0: Use upwind from negative to positive
 */
__device__ inline double godunovGradientWENO5(const double* phi, int i, int j,
                                               double sign_phi0, double dx, double dy,
                                               int nx_total) {
    // Use WENO-5 gradient magnitude from weno5.cuh
    return weno5_gradient_magnitude(phi, i, j, sign_phi0, dx, dy, nx_total);
}

//=============================================================================
// Stability Constraint Check (Eq. 18)
//=============================================================================

/**
 * @brief Check if the cell and its crossing neighbors satisfy the stability constraint C^v
 *
 * The constraint C^v requires that sgn(φ_k) = sgn(φ₀_k) for all k in the stencil.
 * This prevents instability when the interface crosses a cell during reinitialization.
 *
 * @param phi_n Current level-set field
 * @param phi_0 Original level-set field
 * @param i, j Cell indices
 * @param interface_flag Flags indicating which neighbors cross the interface
 * @param nx_total Grid size
 * @return true if constraint is satisfied, false otherwise
 */
__device__ inline bool checkStabilityConstraint(const double* phi_n,
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
// HCR-2 Reinitialization Step Kernel
//=============================================================================

/**
 * @brief Single pseudo-time step of HCR-2 reinitialization
 *
 * Solves (Eq. 15): ∂φ/∂τ + S(φ₀)(|∇φ| - 1) = β * F
 *
 * where F is the HCR-2 forcing term (Eq. 21b):
 *   F = (1/Δx) * (r̃ * Σφ_neighbors - φ)
 *
 * The forcing is only applied within the constraint subspace C^v (Eq. 18).
 *
 * @param phi_n Current level-set field
 * @param phi_new Updated level-set field
 * @param phi_0 Original level-set field (for sign computation)
 * @param r_tilde Pre-computed r̃ values
 * @param interface_flag Interface-adjacent cell flags
 * @param dtau Pseudo-time step
 * @param params Simulation parameters
 */
__global__ void reinitStep(const double* __restrict__ phi_n,
                           double* __restrict__ phi_new,
                           const double* __restrict__ phi_0,
                           const double* __restrict__ r_tilde,
                           const int* __restrict__ interface_flag,
                           double dtau,
                           SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process interior points
    if (i < params.nghost || i >= params.nx + params.nghost ||
        j < params.nghost || j >= params.ny + params.nghost) {
        return;
    }

    int index = idx(i, j, params.nx_total);

    double phi = phi_n[index];
    double phi0 = phi_0[index];

    // Smoothed sign function
    double S = smoothSign(phi0, params.dx);

    // Compute gradient magnitude using WENO-5 with Godunov Hamiltonian
    double grad_mag = godunovGradientWENO5(phi_n, i, j, S, params.dx, params.dy, params.nx_total);

    // Standard reinitialization term: -S(φ₀)(|∇φ| - 1)
    double reinit_term = -S * (grad_mag - 1.0);

    // HCR-2 forcing term (Eq. 21b)
    double forcing = 0.0;
    int flag = interface_flag[index];

    if (flag != 0) {
        // Check stability constraint C^v (Eq. 18)
        bool stable = checkStabilityConstraint(phi_n, phi_0, i, j, flag, params.nx_total);

        if (stable) {
            // Compute sum of neighboring φ values (only those with interface crossings)
            double sum_phi_neighbors = 0.0;

            if (flag & 1) {  // x-minus
                sum_phi_neighbors += phi_n[idx(i - 1, j, params.nx_total)];
            }
            if (flag & 2) {  // x-plus
                sum_phi_neighbors += phi_n[idx(i + 1, j, params.nx_total)];
            }
            if (flag & 4) {  // y-minus
                sum_phi_neighbors += phi_n[idx(i, j - 1, params.nx_total)];
            }
            if (flag & 8) {  // y-plus
                sum_phi_neighbors += phi_n[idx(i, j + 1, params.nx_total)];
            }

            // HCR-2 forcing (Eq. 21b): F = (1/Δx) * (r̃ * Σφ_neighbors - φ)
            double r = r_tilde[index];
            forcing = params.reinit_beta * (r * sum_phi_neighbors - phi) / params.dx;
        }
    }

    // Update: φ^{n+1} = φ^n + Δτ * (reinit_term + forcing)
    phi_new[index] = phi + dtau * (reinit_term + forcing);
}

//=============================================================================
// Full Reinitialization Procedure (Host Function)
//=============================================================================

/**
 * @brief Perform complete HCR-2 reinitialization
 *
 * @param d_phi Current level-set field (will be updated in place)
 * @param d_phi_temp Temporary storage
 * @param d_phi_0 Copy of original field before reinitialization
 * @param d_r_tilde Pre-computed r̃ values
 * @param d_interface_flag Interface-adjacent cell flags
 * @param params Simulation parameters
 * @param applyBoundary Pointer to boundary condition function
 */
void reinitialize(double* d_phi, double* d_phi_temp, double* d_phi_0,
                  double* d_r_tilde, int* d_interface_flag,
                  SimParams& params,
                  void (*applyBoundary)(double*, SimParams&)) {

    if (!params.enable_reinit) return;

    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    int size = params.nx_total * params.ny_total;

    // Copy current field to phi_0 for reference
    CUDA_CHECK(cudaMemcpy(d_phi_0, d_phi, size * sizeof(double), cudaMemcpyDeviceToDevice));

    // Pre-compute interface crossings and r̃ values
    computeInterfaceCrossings<<<grid, block>>>(d_phi_0, d_r_tilde, d_interface_flag, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pseudo-time stepping
    double dtau = params.reinit_dtau;

    for (int iter = 0; iter < params.reinit_iterations; iter++) {
        // Perform one reinitialization step
        reinitStep<<<grid, block>>>(d_phi, d_phi_temp, d_phi_0,
                                     d_r_tilde, d_interface_flag,
                                     dtau, params);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers
        double* temp = d_phi;
        d_phi = d_phi_temp;
        d_phi_temp = temp;

        // Apply boundary conditions
        applyBoundary(d_phi, params);
    }

    // If odd number of iterations, copy back to original
    if (params.reinit_iterations % 2 == 1) {
        CUDA_CHECK(cudaMemcpy(d_phi_temp, d_phi, size * sizeof(double), cudaMemcpyDeviceToDevice));
    }
}

/**
 * @brief Alternative reinitialization interface that handles pointer swapping
 */
void reinitializeWithSwap(double** d_phi_ptr, double** d_phi_temp_ptr, double* d_phi_0,
                          double* d_r_tilde, int* d_interface_flag,
                          SimParams& params,
                          void (*applyBoundary)(double*, SimParams&)) {

    if (!params.enable_reinit) return;

    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    int size = params.nx_total * params.ny_total;

    double* d_phi = *d_phi_ptr;
    double* d_phi_temp = *d_phi_temp_ptr;

    // Copy current field to phi_0 for reference
    CUDA_CHECK(cudaMemcpy(d_phi_0, d_phi, size * sizeof(double), cudaMemcpyDeviceToDevice));

    // Pre-compute interface crossings and r̃ values
    computeInterfaceCrossings<<<grid, block>>>(d_phi_0, d_r_tilde, d_interface_flag, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pseudo-time stepping
    double dtau = params.reinit_dtau;

    for (int iter = 0; iter < params.reinit_iterations; iter++) {
        // Perform one reinitialization step
        reinitStep<<<grid, block>>>(d_phi, d_phi_temp, d_phi_0,
                                     d_r_tilde, d_interface_flag,
                                     dtau, params);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers
        double* temp = d_phi;
        d_phi = d_phi_temp;
        d_phi_temp = temp;

        // Apply boundary conditions
        applyBoundary(d_phi, params);
    }

    // Update output pointers
    *d_phi_ptr = d_phi;
    *d_phi_temp_ptr = d_phi_temp;
}

#endif // REINITIALIZATION_CUH
