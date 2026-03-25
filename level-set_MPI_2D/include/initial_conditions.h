/**
 * @file initial_conditions.h
 * @brief Initial condition generators for level-set field (MPI version)
 *
 * This module provides various initial condition setups including:
 * - Pyramid (upper half of diamond) for advection test
 * - Circle for convergence tests
 * - Zalesak's slotted disk for rotation tests
 *
 * All initial conditions produce signed distance functions where:
 * - G < 0 inside the interface (burned region for flame)
 * - G > 0 outside the interface (unburned region)
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

// Pyramid apex position
constexpr double PYRAMID_CENTER_X = 0.5;
constexpr double PYRAMID_CENTER_Y = 1.0;
constexpr double PYRAMID_HALF_WIDTH = 0.5;

// Circle parameters
constexpr double CIRCLE_CENTER_X = 0.25;
constexpr double CIRCLE_CENTER_Y = 0.5;
constexpr double CIRCLE_RADIUS = 0.15;

// Zalesak's slotted disk parameters
constexpr double ZALESAK_CENTER_X = 0.5;
constexpr double ZALESAK_CENTER_Y = 0.75;
constexpr double ZALESAK_RADIUS = 0.15;
constexpr double ZALESAK_SLOT_WIDTH = 0.05;
constexpr double ZALESAK_SLOT_LENGTH = 0.25;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================
// Signed Distance Function Computations
//=============================================================================

/**
 * @brief Compute signed distance to a pyramid shape
 */
inline double signedDistancePyramid(double x, double y,
                                    double cx, double cy_base, double r) {
    (void)cy_base;
    (void)r;
    double px = fabs(x - cx);
    double distance = (2.0 * px + y - 1.0) / sqrt(5.0);
    return distance;
}

/**
 * @brief Compute signed distance to a circle
 */
inline double signedDistanceCircle(double x, double y,
                                   double cx, double cy, double r) {
    double dx = x - cx;
    double dy = y - cy;
    return sqrt(dx * dx + dy * dy) - r;
}

/**
 * @brief Compute signed distance to a square
 */
inline double signedDistanceSquare(double x, double y,
                                   double cx, double cy, double half_size) {
    double dx = fabs(x - cx) - half_size;
    double dy = fabs(y - cy) - half_size;

    double outside = sqrt(fmax(dx, 0.0) * fmax(dx, 0.0) + fmax(dy, 0.0) * fmax(dy, 0.0));
    double inside = fmin(fmax(dx, dy), 0.0);

    return outside + inside;
}

/**
 * @brief Compute signed distance to Zalesak's slotted disk
 */
inline double signedDistanceZalesak(double x, double y,
                                    double cx, double cy, double r,
                                    double slot_width, double slot_length) {
    double dx = x - cx;
    double dy = y - cy;
    double dist_disk = sqrt(dx * dx + dy * dy) - r;

    double slot_half_width = slot_width / 2.0;
    double slot_bottom = cy - r;
    double slot_top = slot_bottom + slot_length;

    double slot_dx = fabs(x - cx) - slot_half_width;
    double slot_dy = fmax(slot_bottom - y, y - slot_top);

    double dist_slot;
    if (slot_dx < 0.0 && slot_dy < 0.0) {
        dist_slot = fmax(slot_dx, slot_dy);
    } else if (slot_dx < 0.0) {
        dist_slot = slot_dy;
    } else if (slot_dy < 0.0) {
        dist_slot = slot_dx;
    } else {
        dist_slot = sqrt(slot_dx * slot_dx + slot_dy * slot_dy);
    }

    return fmax(dist_disk, -dist_slot);
}

//=============================================================================
// Initialization Functions (MPI-aware)
//=============================================================================

/**
 * @brief Initialize G field with pyramid shape (local portion)
 */
inline void initPyramid(double* G, SimParams& params,
                        double cx, double cy, double r) {
    int nghost = params.nghost;
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    for (int j = 0; j < local_ny_total; j++) {
        for (int i = 0; i < nx_total; i++) {
            double x, y;
            indexToCoord(i, j, params.x_min, params.y_min,
                        params.dx, params.dy, nghost,
                        params.y_start, x, y);

            int index = idx(i, j, nx_total);
            G[index] = signedDistancePyramid(x, y, cx, cy, r);
        }
    }
}

/**
 * @brief Initialize G field with circle shape (local portion)
 */
inline void initCircle(double* G, SimParams& params,
                       double cx, double cy, double r) {
    int nghost = params.nghost;
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    for (int j = 0; j < local_ny_total; j++) {
        for (int i = 0; i < nx_total; i++) {
            double x, y;
            indexToCoord(i, j, params.x_min, params.y_min,
                        params.dx, params.dy, nghost,
                        params.y_start, x, y);

            int index = idx(i, j, nx_total);
            G[index] = signedDistanceCircle(x, y, cx, cy, r);
        }
    }
}

/**
 * @brief Initialize G field with Zalesak's slotted disk (local portion)
 */
inline void initZalesak(double* G, SimParams& params,
                        double cx, double cy, double r,
                        double slot_width, double slot_length) {
    int nghost = params.nghost;
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    for (int j = 0; j < local_ny_total; j++) {
        for (int i = 0; i < nx_total; i++) {
            double x, y;
            indexToCoord(i, j, params.x_min, params.y_min,
                        params.dx, params.dy, nghost,
                        params.y_start, x, y);

            int index = idx(i, j, nx_total);
            G[index] = signedDistanceZalesak(x, y, cx, cy, r, slot_width, slot_length);
        }
    }
}

/**
 * @brief Initialize constant velocity field (local portion)
 */
inline void initConstantVelocity(double* u, double* v, SimParams& params) {
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    for (int j = 0; j < local_ny_total; j++) {
        for (int i = 0; i < nx_total; i++) {
            int index = idx(i, j, nx_total);
            u[index] = params.u_const;
            v[index] = params.v_const;
        }
    }
}

/**
 * @brief Initialize rotating velocity field (local portion)
 */
inline void initRotatingVelocity(double* u, double* v, SimParams& params,
                                 double rot_cx, double rot_cy, double omega) {
    int nghost = params.nghost;
    int nx_total = params.nx_total;
    int local_ny_total = params.local_ny_total;

    for (int j = 0; j < local_ny_total; j++) {
        for (int i = 0; i < nx_total; i++) {
            double x, y;
            indexToCoord(i, j, params.x_min, params.y_min,
                        params.dx, params.dy, nghost,
                        params.y_start, x, y);

            int index = idx(i, j, nx_total);
            u[index] = -omega * (y - rot_cy);
            v[index] = omega * (x - rot_cx);
        }
    }
}

//=============================================================================
// Test Case Initialization Functions
//=============================================================================

/**
 * @brief Initialize pyramid test case
 */
inline void initPyramidTestCase(double* G, double* u, double* v, SimParams& params) {
    double cx = 0.5;
    double cy_apex = 1.0;
    double r = 0.5;

    initPyramid(G, params, cx, cy_apex, r);
    initConstantVelocity(u, v, params);
}

/**
 * @brief Initialize circle test case
 */
inline void initCircleTestCase(double* G, double* u, double* v, SimParams& params) {
    double cx = params.x_min + CIRCLE_CENTER_X * (params.x_max - params.x_min);
    double cy = params.y_min + CIRCLE_CENTER_Y * (params.y_max - params.y_min);
    double r = CIRCLE_RADIUS * (params.x_max - params.x_min);

    initCircle(G, params, cx, cy, r);
    initConstantVelocity(u, v, params);
}

/**
 * @brief Initialize Zalesak's slotted disk test case
 */
inline void initZalesakTestCase(double* G, double* u, double* v, SimParams& params) {
    double disk_cx = ZALESAK_CENTER_X;
    double disk_cy = ZALESAK_CENTER_Y;
    double disk_r = ZALESAK_RADIUS;
    double slot_width = ZALESAK_SLOT_WIDTH;
    double slot_length = ZALESAK_SLOT_LENGTH;

    double rot_cx = 0.5;
    double rot_cy = 0.5;
    double omega = 2.0 * M_PI / params.t_final;

    initZalesak(G, params, disk_cx, disk_cy, disk_r, slot_width, slot_length);
    initRotatingVelocity(u, v, params, rot_cx, rot_cy, omega);

    if (params.rank == 0) {
        printf("Zalesak Disk Test Case:\n");
        printf("  Disk center: (%.2f, %.2f), radius: %.2f\n", disk_cx, disk_cy, disk_r);
        printf("  Slot: width=%.2f, length=%.2f\n", slot_width, slot_length);
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
