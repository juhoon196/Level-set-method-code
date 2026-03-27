/**
 * @file boundary.h
 * @brief Boundary condition implementations for 3D G-equation solver (MPI version)
 *
 * - Periodic boundary conditions (x- and z-directions, local)
 * - Zero-gradient (Neumann) boundary conditions (y-direction domain edges)
 * - MPI ghost cell exchange between neighboring processes (y-direction)
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
// MPI Ghost Cell Exchange (Y-direction, 3D)
//=============================================================================

/**
 * @brief Exchange ghost cells between MPI neighbors in Y-direction (3D)
 *
 * Each process sends its boundary YZ-slabs to neighbors and receives ghost slabs.
 * A Y-slab consists of nghost rows, each of size nx_total * nz_total.
 */
inline void exchangeGhostCells3D(double* G, SimParams& params) {
    int nx_total = params.nx_total;
    int nz_total = params.nz_total;
    int nghost = params.nghost;
    int local_ny = params.local_ny;
    int local_ny_total = params.local_ny_total;

    // Size of one Y-row in 3D: nx_total * nz_total
    // But our memory layout is (k * ny_total + j) * nx_total + i
    // So a contiguous y-slab of nghost rows is NOT contiguous in memory.
    // We need to pack/unpack buffers.

    int slab_size = nghost * nx_total * nz_total;
    double* send_buf_down = new double[slab_size];
    double* recv_buf_down = new double[slab_size];
    double* send_buf_up   = new double[slab_size];
    double* recv_buf_up   = new double[slab_size];

    // Pack: send first nghost interior rows downward
    // Interior rows: j = nghost .. nghost + local_ny - 1
    // Send down: j = nghost .. nghost + nghost - 1
    if (params.neighbor_below != MPI_PROC_NULL) {
        int pos = 0;
        for (int k = 0; k < nz_total; k++) {
            for (int jj = 0; jj < nghost; jj++) {
                int j = nghost + jj;
                for (int i = 0; i < nx_total; i++) {
                    send_buf_down[pos++] = G[idx3(i, j, k, nx_total, local_ny_total)];
                }
            }
        }
    }

    // Pack: send last nghost interior rows upward
    // Send up: j = nghost + local_ny - nghost .. nghost + local_ny - 1
    if (params.neighbor_above != MPI_PROC_NULL) {
        int pos = 0;
        for (int k = 0; k < nz_total; k++) {
            for (int jj = 0; jj < nghost; jj++) {
                int j = nghost + local_ny - nghost + jj;
                for (int i = 0; i < nx_total; i++) {
                    send_buf_up[pos++] = G[idx3(i, j, k, nx_total, local_ny_total)];
                }
            }
        }
    }

    MPI_Request requests[4];
    MPI_Status statuses[4];
    int num_requests = 0;

    // Send to below, receive from below
    if (params.neighbor_below != MPI_PROC_NULL) {
        MPI_Isend(send_buf_down, slab_size, MPI_DOUBLE,
                  params.neighbor_below, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(recv_buf_down, slab_size, MPI_DOUBLE,
                  params.neighbor_below, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    // Send to above, receive from above
    if (params.neighbor_above != MPI_PROC_NULL) {
        MPI_Isend(send_buf_up, slab_size, MPI_DOUBLE,
                  params.neighbor_above, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(recv_buf_up, slab_size, MPI_DOUBLE,
                  params.neighbor_above, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    if (num_requests > 0) {
        MPI_Waitall(num_requests, requests, statuses);
    }

    // Unpack: bottom ghost cells (j = 0 .. nghost - 1)
    if (params.neighbor_below != MPI_PROC_NULL) {
        int pos = 0;
        for (int k = 0; k < nz_total; k++) {
            for (int jj = 0; jj < nghost; jj++) {
                int j = jj;
                for (int i = 0; i < nx_total; i++) {
                    G[idx3(i, j, k, nx_total, local_ny_total)] = recv_buf_down[pos++];
                }
            }
        }
    }

    // Unpack: top ghost cells (j = nghost + local_ny .. local_ny_total - 1)
    if (params.neighbor_above != MPI_PROC_NULL) {
        int pos = 0;
        for (int k = 0; k < nz_total; k++) {
            for (int jj = 0; jj < nghost; jj++) {
                int j = nghost + local_ny + jj;
                for (int i = 0; i < nx_total; i++) {
                    G[idx3(i, j, k, nx_total, local_ny_total)] = recv_buf_up[pos++];
                }
            }
        }
    }

    delete[] send_buf_down;
    delete[] recv_buf_down;
    delete[] send_buf_up;
    delete[] recv_buf_up;
}

//=============================================================================
// Periodic Boundary Conditions (X-direction, 3D)
//=============================================================================

inline void applyPeriodicBC_X_3D(double* G, int nx, int ny_local, int nz_total,
                                  int nghost, int nx_total, int ny_local_total) {
    for (int k = 0; k < nz_total; k++) {
        for (int j = 0; j < ny_local + 2 * nghost; j++) {
            for (int g = 0; g < nghost; g++) {
                // Left ghost
                int src_i = nx + g;
                int dst_i = g;
                G[idx3(dst_i, j, k, nx_total, ny_local_total)] =
                    G[idx3(src_i, j, k, nx_total, ny_local_total)];

                // Right ghost
                src_i = nghost + g;
                dst_i = nx + nghost + g;
                G[idx3(dst_i, j, k, nx_total, ny_local_total)] =
                    G[idx3(src_i, j, k, nx_total, ny_local_total)];
            }
        }
    }
}

//=============================================================================
// Periodic Boundary Conditions (Z-direction, 3D)
//=============================================================================

inline void applyPeriodicBC_Z_3D(double* G, int nx_total, int ny_local, int nz,
                                  int nghost, int ny_local_total) {
    for (int j = 0; j < ny_local + 2 * nghost; j++) {
        for (int i = 0; i < nx_total; i++) {
            for (int g = 0; g < nghost; g++) {
                // Back ghost (k = 0..nghost-1)
                int src_k = nz + g;        // from front interior
                int dst_k = g;
                G[idx3(i, j, dst_k, nx_total, ny_local_total)] =
                    G[idx3(i, j, src_k, nx_total, ny_local_total)];

                // Front ghost (k = nz+nghost..nz+2*nghost-1)
                src_k = nghost + g;        // from back interior
                dst_k = nz + nghost + g;
                G[idx3(i, j, dst_k, nx_total, ny_local_total)] =
                    G[idx3(i, j, src_k, nx_total, ny_local_total)];
            }
        }
    }
}

//=============================================================================
// Zero-Gradient (Neumann) Boundary Conditions (Y-direction, 3D)
//=============================================================================

inline void applyZeroGradientBC_Y_3D(double* G, int nx_total, int ny_local, int nz_total,
                                      int nghost, int ny_local_total,
                                      bool is_bottom, bool is_top) {
    for (int k = 0; k < nz_total; k++) {
        for (int i = 0; i < nx_total; i++) {
            if (is_bottom) {
                double val = G[idx3(i, nghost, k, nx_total, ny_local_total)];
                for (int g = 0; g < nghost; g++) {
                    G[idx3(i, g, k, nx_total, ny_local_total)] = val;
                }
            }
            if (is_top) {
                double val = G[idx3(i, ny_local + nghost - 1, k, nx_total, ny_local_total)];
                for (int g = 0; g < nghost; g++) {
                    G[idx3(i, ny_local + nghost + g, k, nx_total, ny_local_total)] = val;
                }
            }
        }
    }
}

//=============================================================================
// Zero-Gradient BC (Z-direction, 3D)
//=============================================================================

inline void applyZeroGradientBC_Z_3D(double* G, int nx_total, int ny_local, int nz,
                                      int nghost, int ny_local_total) {
    for (int j = 0; j < ny_local + 2 * nghost; j++) {
        for (int i = 0; i < nx_total; i++) {
            // Back ghost
            double back_val = G[idx3(i, j, nghost, nx_total, ny_local_total)];
            for (int g = 0; g < nghost; g++) {
                G[idx3(i, j, g, nx_total, ny_local_total)] = back_val;
            }

            // Front ghost
            double front_val = G[idx3(i, j, nz + nghost - 1, nx_total, ny_local_total)];
            for (int g = 0; g < nghost; g++) {
                G[idx3(i, j, nz + nghost + g, nx_total, ny_local_total)] = front_val;
            }
        }
    }
}

//=============================================================================
// Combined Boundary Condition Application (3D)
//=============================================================================

/**
 * @brief Apply boundary conditions to G field (MPI 3D version)
 *
 * Configuration (matching GPU 3D):
 * - X-direction: Periodic
 * - Y-direction: Periodic (via MPI ghost exchange with wrap-around)
 * - Z-direction: Periodic
 */
inline void applyBoundaryConditions3D(double* G, SimParams& params) {
    // MPI ghost exchange in Y (periodic: first/last processes wrap around)
    exchangeGhostCells3D(G, params);

    // Periodic BC in X
    applyPeriodicBC_X_3D(G, params.nx, params.local_ny, params.nz_total,
                          params.nghost, params.nx_total, params.local_ny_total);

    // Periodic BC in Z
    applyPeriodicBC_Z_3D(G, params.nx_total, params.local_ny, params.nz,
                          params.nghost, params.local_ny_total);
}

/**
 * @brief Apply boundary conditions to velocity field components (3D)
 */
inline void applyVelocityBoundaryConditions3D(double* u, double* v, double* w,
                                               SimParams& params) {
    applyBoundaryConditions3D(u, params);
    applyBoundaryConditions3D(v, params);
    applyBoundaryConditions3D(w, params);
}

#endif // BOUNDARY_H
