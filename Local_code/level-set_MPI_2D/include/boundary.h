/**
 * @file boundary.h
 * @brief Boundary condition implementations for the G-equation solver (MPI version)
 *
 * This module provides various boundary condition treatments:
 * - Periodic boundary conditions (x-direction)
 * - Zero-gradient (Neumann) boundary conditions (y-direction edges)
 * - MPI ghost cell exchange between neighboring processes
 */

#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <mpi.h>
#include "config.h"

//=============================================================================
// Boundary Condition Types
//=============================================================================

enum BoundaryType {
    BC_PERIODIC,
    BC_ZERO_GRADIENT,
    BC_EXTRAPOLATION,
    BC_DIRICHLET
};

//=============================================================================
// MPI Ghost Cell Exchange
//=============================================================================

/**
 * @brief Exchange ghost cells between MPI neighbors in Y-direction
 *
 * Each process sends its boundary rows to neighbors and receives ghost cells.
 * Uses non-blocking MPI for efficiency.
 *
 * @param G Local G field array
 * @param params Simulation parameters (includes MPI info)
 */
inline void exchangeGhostCells(double* G, SimParams& params) {
    int nx_total = params.nx_total;
    int nghost = params.nghost;
    int local_ny = params.local_ny;
    int row_size = nx_total;

    MPI_Request requests[4];
    MPI_Status statuses[4];
    int num_requests = 0;

    // Send/receive buffer pointers
    // Bottom ghost cells: rows [0, nghost)
    // Top rows to send down: rows [nghost, 2*nghost) (first nghost interior rows)
    // Bottom rows to send up: rows [local_ny, local_ny+nghost) (last nghost interior rows)
    // Top ghost cells: rows [local_ny+nghost, local_ny+2*nghost)

    // Send to below, receive from below
    if (params.neighbor_below != MPI_PROC_NULL) {
        // Send first nghost interior rows to process below
        double* send_buf = &G[nghost * row_size];
        double* recv_buf = &G[0];  // Bottom ghost cells

        MPI_Isend(send_buf, nghost * row_size, MPI_DOUBLE,
                  params.neighbor_below, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(recv_buf, nghost * row_size, MPI_DOUBLE,
                  params.neighbor_below, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    // Send to above, receive from above
    if (params.neighbor_above != MPI_PROC_NULL) {
        // Send last nghost interior rows to process above
        double* send_buf = &G[local_ny * row_size];  // local_ny = local interior rows
        double* recv_buf = &G[(local_ny + nghost) * row_size];  // Top ghost cells

        MPI_Isend(send_buf, nghost * row_size, MPI_DOUBLE,
                  params.neighbor_above, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(recv_buf, nghost * row_size, MPI_DOUBLE,
                  params.neighbor_above, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    // Wait for all communications to complete
    if (num_requests > 0) {
        MPI_Waitall(num_requests, requests, statuses);
    }
}

//=============================================================================
// Periodic Boundary Conditions (X-direction)
//=============================================================================

/**
 * @brief Apply periodic boundary conditions in x-direction
 *
 * For WENO-5 with 3 ghost cells on each side:
 * - Left ghost cells copy from right interior cells
 * - Right ghost cells copy from left interior cells
 */
inline void applyPeriodicBC_X(double* G, int nx, int ny_local, int nghost, int nx_total) {
    for (int j = 0; j < ny_local + 2 * nghost; j++) {
        // Left ghost cells (i = 0, 1, 2)
        for (int g = 0; g < nghost; g++) {
            int src_i = nx + g;  // Source from right interior
            int dst_i = g;       // Destination left ghost
            G[idx(dst_i, j, nx_total)] = G[idx(src_i, j, nx_total)];
        }

        // Right ghost cells (i = nx+nghost, nx+nghost+1, nx+nghost+2)
        for (int g = 0; g < nghost; g++) {
            int src_i = nghost + g;         // Source from left interior
            int dst_i = nx + nghost + g;    // Destination right ghost
            G[idx(dst_i, j, nx_total)] = G[idx(src_i, j, nx_total)];
        }
    }
}

//=============================================================================
// Zero-Gradient (Neumann) Boundary Conditions
//=============================================================================

/**
 * @brief Apply zero-gradient BC at top/bottom domain boundaries
 *
 * This is only applied at the global domain boundaries (first and last processes)
 * Ghost cells are set equal to the nearest interior cell
 */
inline void applyZeroGradientBC_Y(double* G, int nx, int ny_local, int nghost, int nx_total,
                                   bool is_bottom, bool is_top) {
    for (int i = 0; i < nx + 2 * nghost; i++) {
        if (is_bottom) {
            // Get boundary value (first interior row)
            double bottom_val = G[idx(i, nghost, nx_total)];
            // Fill bottom ghost cells
            for (int g = 0; g < nghost; g++) {
                G[idx(i, g, nx_total)] = bottom_val;
            }
        }

        if (is_top) {
            // Get boundary value (last interior row)
            double top_val = G[idx(i, ny_local + nghost - 1, nx_total)];
            // Fill top ghost cells
            for (int g = 0; g < nghost; g++) {
                G[idx(i, ny_local + nghost + g, nx_total)] = top_val;
            }
        }
    }
}

/**
 * @brief Apply zero-gradient BC in x-direction (left and right)
 */
inline void applyZeroGradientBC_X(double* G, int nx, int ny_local, int nghost, int nx_total) {
    for (int j = 0; j < ny_local + 2 * nghost; j++) {
        // Get boundary values
        double left_val = G[idx(nghost, j, nx_total)];             // First interior cell
        double right_val = G[idx(nx + nghost - 1, j, nx_total)];   // Last interior cell

        // Fill left ghost cells
        for (int g = 0; g < nghost; g++) {
            G[idx(g, j, nx_total)] = left_val;
        }

        // Fill right ghost cells
        for (int g = 0; g < nghost; g++) {
            G[idx(nx + nghost + g, j, nx_total)] = right_val;
        }
    }
}

//=============================================================================
// Extrapolation Boundary Conditions
//=============================================================================

/**
 * @brief Apply linear extrapolation BC at global Y boundaries
 */
inline void applyExtrapolationBC_Y(double* G, int nx, int ny_local, int nghost, int nx_total,
                                    bool is_bottom, bool is_top) {
    for (int i = 0; i < nx + 2 * nghost; i++) {
        if (is_bottom) {
            double val0 = G[idx(i, nghost, nx_total)];
            double val1 = G[idx(i, nghost + 1, nx_total)];
            double slope_bottom = val0 - val1;

            for (int g = 0; g < nghost; g++) {
                G[idx(i, nghost - 1 - g, nx_total)] = val0 + (g + 1) * slope_bottom;
            }
        }

        if (is_top) {
            double val0 = G[idx(i, ny_local + nghost - 1, nx_total)];
            double val1 = G[idx(i, ny_local + nghost - 2, nx_total)];
            double slope_top = val0 - val1;

            for (int g = 0; g < nghost; g++) {
                G[idx(i, ny_local + nghost + g, nx_total)] = val0 + (g + 1) * slope_top;
            }
        }
    }
}

//=============================================================================
// Combined Boundary Condition Application
//=============================================================================

/**
 * @brief Apply boundary conditions to G field (MPI version)
 *
 * Default configuration:
 * - X-direction: Periodic
 * - Y-direction: Zero-gradient (Neumann) at global domain boundaries
 * - Interior Y boundaries: MPI ghost exchange
 *
 * @param G Local G field (device pointer)
 * @param params Simulation parameters
 */
inline void applyBoundaryConditions(double* G, SimParams& params) {
    // First, exchange ghost cells with MPI neighbors
    exchangeGhostCells(G, params);

    // Apply periodic BC in x-direction (local operation)
    applyPeriodicBC_X(G, params.nx, params.local_ny, params.nghost, params.nx_total);

    // Apply zero-gradient BC at global Y boundaries
    bool is_bottom = (params.neighbor_below == MPI_PROC_NULL);
    bool is_top = (params.neighbor_above == MPI_PROC_NULL);
    applyZeroGradientBC_Y(G, params.nx, params.local_ny, params.nghost, params.nx_total,
                          is_bottom, is_top);
}

/**
 * @brief Apply fully periodic boundary conditions (both directions)
 *
 * Note: Y-direction periodic requires wrap-around communication between
 * first and last processes.
 */
inline void applyPeriodicBoundaryConditions(double* G, SimParams& params) {
    // Exchange ghost cells with MPI neighbors
    exchangeGhostCells(G, params);

    // Apply periodic BC in x-direction
    applyPeriodicBC_X(G, params.nx, params.local_ny, params.nghost, params.nx_total);

    // For fully periodic: first and last process need to communicate
    // This is handled by the MPI topology if set up with periodic boundaries
    // For now, we assume zero-gradient at global Y boundaries if not using periodic
}

//=============================================================================
// Velocity Field Boundary Conditions
//=============================================================================

/**
 * @brief Apply boundary conditions to velocity field components
 *
 * @param u X-velocity component
 * @param v Y-velocity component
 * @param params Simulation parameters
 */
inline void applyVelocityBoundaryConditions(double* u, double* v, SimParams& params) {
    // Exchange ghost cells for both velocity components
    exchangeGhostCells(u, params);
    exchangeGhostCells(v, params);

    // Apply periodic BC in x-direction
    applyPeriodicBC_X(u, params.nx, params.local_ny, params.nghost, params.nx_total);
    applyPeriodicBC_X(v, params.nx, params.local_ny, params.nghost, params.nx_total);

    // Apply zero-gradient at global Y boundaries
    bool is_bottom = (params.neighbor_below == MPI_PROC_NULL);
    bool is_top = (params.neighbor_above == MPI_PROC_NULL);
    applyZeroGradientBC_Y(u, params.nx, params.local_ny, params.nghost, params.nx_total,
                          is_bottom, is_top);
    applyZeroGradientBC_Y(v, params.nx, params.local_ny, params.nghost, params.nx_total,
                          is_bottom, is_top);
}

#endif // BOUNDARY_H
