/**
 * @file initial_conditions.h
 * @brief Initial condition generators for 3D level-set field (MPI version)
 *
 * Provides:
 * - Sphere for advection/convergence tests
 * - Zalesak's slotted sphere for rotation tests
 *
 * Sign convention:
 * - G < 0 inside the interface
 * - G > 0 outside the interface
 * - G = 0 at the interface
 */

#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include <cmath>
#include <cstdio>
#include "config.h"

//=============================================================================
// Shape Parameters
//=============================================================================

// Sphere parameters
constexpr double SPHERE_CENTER_X = 0.25;
constexpr double SPHERE_CENTER_Y = 0.5;
constexpr double SPHERE_CENTER_Z = 0.5;
constexpr double SPHERE_RADIUS = 0.15;

// Zalesak's slotted sphere parameters
constexpr double ZALESAK3D_CENTER_X = 0.5;
constexpr double ZALESAK3D_CENTER_Y = 0.75;
constexpr double ZALESAK3D_CENTER_Z = 0.5;
constexpr double ZALESAK3D_RADIUS = 0.15;
constexpr double ZALESAK3D_SLOT_WIDTH = 0.05;
constexpr double ZALESAK3D_SLOT_DEPTH = 0.05;
constexpr double ZALESAK3D_SLOT_LENGTH = 0.25;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================
// Signed Distance Functions (3D)
//=============================================================================

/**
 * @brief Compute signed distance to a sphere
 */
inline double signedDistanceSphere(double x, double y, double z,
                                   double cx, double cy, double cz, double r) {
    double dx = x - cx;
    double dy = y - cy;
    double dz = z - cz;
    return sqrt(dx * dx + dy * dy + dz * dz) - r;
}

/**
 * @brief Compute signed distance to a cube
 */
inline double signedDistanceCube(double x, double y, double z,
                                 double cx, double cy, double cz, double half_size) {
    double dx = fabs(x - cx) - half_size;
    double dy = fabs(y - cy) - half_size;
    double dz = fabs(z - cz) - half_size;

    double outside = sqrt(fmax(dx, 0.0) * fmax(dx, 0.0) +
                          fmax(dy, 0.0) * fmax(dy, 0.0) +
                          fmax(dz, 0.0) * fmax(dz, 0.0));
    double inside = fmin(fmax(dx, fmax(dy, dz)), 0.0);

    return outside + inside;
}

/**
 * @brief Compute signed distance to Zalesak's slotted sphere (3D)
 */
inline double signedDistanceZalesak3D(double x, double y, double z,
                                      double cx, double cy, double cz, double r,
                                      double slot_width, double slot_depth, double slot_length) {
    double dx = x - cx;
    double dy = y - cy;
    double dz = z - cz;
    double dist_sphere = sqrt(dx * dx + dy * dy + dz * dz) - r;

    double slot_half_w = slot_width / 2.0;
    double slot_half_d = slot_depth / 2.0;
    double slot_bottom = cy - r;
    double slot_top = slot_bottom + slot_length;

    // Slot is a rectangular prism centered at (cx, slot_center_y, cz)
    double sx = fabs(x - cx) - slot_half_w;
    double sz = fabs(z - cz) - slot_half_d;
    double sy = fmax(slot_bottom - y, y - slot_top);

    double dist_slot;
    double mx = fmax(sx, 0.0);
    double my = fmax(sy, 0.0);
    double mz = fmax(sz, 0.0);

    if (sx < 0.0 && sy < 0.0 && sz < 0.0) {
        dist_slot = fmax(sx, fmax(sy, sz));
    } else {
        dist_slot = sqrt(mx * mx + my * my + mz * mz);
    }

    return fmax(dist_sphere, -dist_slot);
}

//=============================================================================
// Initialization Functions (MPI-aware, 3D)
//=============================================================================

/**
 * @brief Initialize G field with sphere shape (local portion)
 */
inline void initSphere(double* G, SimParams& params,
                        double cx, double cy, double cz, double r) {
    int nghost = params.nghost;
    int nx_total = params.nx_total;
    int nz_total = params.nz_total;
    int local_ny_total = params.local_ny_total;

    for (int k = 0; k < nz_total; k++) {
        for (int j = 0; j < local_ny_total; j++) {
            for (int i = 0; i < nx_total; i++) {
                double x, y, z;
                indexToCoord3D(i, j, k,
                               params.x_min, params.y_min, params.z_min,
                               params.dx, params.dy, params.dz, nghost,
                               params.y_start, x, y, z);

                int index = idx3(i, j, k, nx_total, local_ny_total);
                G[index] = signedDistanceSphere(x, y, z, cx, cy, cz, r);
            }
        }
    }
}

/**
 * @brief Initialize G field with Zalesak's slotted sphere (local portion)
 */
inline void initZalesak3D(double* G, SimParams& params,
                           double cx, double cy, double cz, double r,
                           double slot_width, double slot_depth, double slot_length) {
    int nghost = params.nghost;
    int nx_total = params.nx_total;
    int nz_total = params.nz_total;
    int local_ny_total = params.local_ny_total;

    for (int k = 0; k < nz_total; k++) {
        for (int j = 0; j < local_ny_total; j++) {
            for (int i = 0; i < nx_total; i++) {
                double x, y, z;
                indexToCoord3D(i, j, k,
                               params.x_min, params.y_min, params.z_min,
                               params.dx, params.dy, params.dz, nghost,
                               params.y_start, x, y, z);

                int index = idx3(i, j, k, nx_total, local_ny_total);
                G[index] = signedDistanceZalesak3D(x, y, z, cx, cy, cz, r,
                                                   slot_width, slot_depth, slot_length);
            }
        }
    }
}

/**
 * @brief Initialize constant velocity field (3D, local portion)
 */
inline void initConstantVelocity3D(double* u, double* v, double* w, SimParams& params) {
    int nx_total = params.nx_total;
    int nz_total = params.nz_total;
    int local_ny_total = params.local_ny_total;

    for (int k = 0; k < nz_total; k++) {
        for (int j = 0; j < local_ny_total; j++) {
            for (int i = 0; i < nx_total; i++) {
                int index = idx3(i, j, k, nx_total, local_ny_total);
                u[index] = params.u_const;
                v[index] = params.v_const;
                w[index] = params.w_const;
            }
        }
    }
}

/**
 * @brief Initialize rotating velocity field (rotation around z-axis, 3D)
 */
inline void initRotatingVelocity3D(double* u, double* v, double* w, SimParams& params,
                                    double rot_cx, double rot_cy, double omega) {
    int nghost = params.nghost;
    int nx_total = params.nx_total;
    int nz_total = params.nz_total;
    int local_ny_total = params.local_ny_total;

    for (int k = 0; k < nz_total; k++) {
        for (int j = 0; j < local_ny_total; j++) {
            for (int i = 0; i < nx_total; i++) {
                double x, y, z;
                indexToCoord3D(i, j, k,
                               params.x_min, params.y_min, params.z_min,
                               params.dx, params.dy, params.dz, nghost,
                               params.y_start, x, y, z);

                int index = idx3(i, j, k, nx_total, local_ny_total);
                // Rotation around z-axis passing through (rot_cx, rot_cy)
                u[index] = -omega * (y - rot_cy);
                v[index] =  omega * (x - rot_cx);
                w[index] = 0.0;
            }
        }
    }
}

//=============================================================================
// Test Case Initialization Functions (3D)
//=============================================================================

/**
 * @brief Initialize sphere advection test case
 */
inline void initSphereTestCase(double* G, double* u, double* v, double* w,
                                SimParams& params) {
    double cx = params.x_min + SPHERE_CENTER_X * (params.x_max - params.x_min);
    double cy = params.y_min + SPHERE_CENTER_Y * (params.y_max - params.y_min);
    double cz = params.z_min + SPHERE_CENTER_Z * (params.z_max - params.z_min);
    double r = SPHERE_RADIUS * (params.x_max - params.x_min);

    initSphere(G, params, cx, cy, cz, r);
    initConstantVelocity3D(u, v, w, params);
}

/**
 * @brief Initialize Zalesak's slotted sphere test case (3D)
 */
inline void initZalesak3DTestCase(double* G, double* u, double* v, double* w,
                                   SimParams& params) {
    double cx = ZALESAK3D_CENTER_X;
    double cy = ZALESAK3D_CENTER_Y;
    double cz = ZALESAK3D_CENTER_Z;
    double r  = ZALESAK3D_RADIUS;
    double sw = ZALESAK3D_SLOT_WIDTH;
    double sd = ZALESAK3D_SLOT_DEPTH;
    double sl = ZALESAK3D_SLOT_LENGTH;

    double rot_cx = 0.5;
    double rot_cy = 0.5;
    double omega = 2.0 * M_PI / params.t_final;

    initZalesak3D(G, params, cx, cy, cz, r, sw, sd, sl);
    initRotatingVelocity3D(u, v, w, params, rot_cx, rot_cy, omega);

    if (params.rank == 0) {
        printf("Zalesak 3D Sphere Test Case:\n");
        printf("  Center: (%.2f, %.2f, %.2f), radius: %.2f\n", cx, cy, cz, r);
        printf("  Slot: width=%.2f, depth=%.2f, length=%.2f\n", sw, sd, sl);
        printf("  Rotation: center=(%.2f, %.2f), omega=%.4f rad/s\n", rot_cx, rot_cy, omega);
        printf("  One full rotation in T=%.2f\n", params.t_final);
    }
}

/**
 * @brief Copy initial G field for error calculation
 */
inline void copyInitialField(const double* G_src, double* G_initial, int size) {
    for (int i = 0; i < size; i++) {
        G_initial[i] = G_src[i];
    }
}

#endif // INITIAL_CONDITIONS_H
