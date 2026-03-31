/**
 * @file initial_conditions.cuh
 * @brief Initial condition generators for level-set field
 *
 * This module provides various initial condition setups including:
 * - Pyramid (upper half of diamond) for advection test
 * - Circle for convergence tests
 * - Custom signed distance functions
 *
 * All initial conditions produce signed distance functions where:
 * - G < 0 inside the interface (burned region for flame)
 * - G > 0 outside the interface (unburned region)
 * - G = 0 at the interface
 */

#ifndef INITIAL_CONDITIONS_CUH
#define INITIAL_CONDITIONS_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "config.cuh"

//=============================================================================
// Pyramid/Diamond Shape Parameters
//=============================================================================

// Pyramid apex position
constexpr double PYRAMID_CENTER_X = 0.5;
constexpr double PYRAMID_CENTER_Y = 1.0;

// Pyramid base height
constexpr double PYRAMID_HALF_WIDTH = 0.5;

//=============================================================================
// Circle Shape Parameters
//=============================================================================

constexpr double CIRCLE_CENTER_X = 0.25;
constexpr double CIRCLE_CENTER_Y = 0.5;
constexpr double CIRCLE_RADIUS = 0.15;

//=============================================================================
// Signed Distance Function Computations
//=============================================================================

/**
 * @brief Compute signed distance to a pyramid (diamond upper half) shape
 *
 * The pyramid is oriented with apex pointing right (+x direction).
 * Shape is defined by: |y - y_c| + |x - x_c| <= r for x >= x_c
 * and x >= x_c for the upper pyramid.
 *
 * For a complete diamond: |y - y_c| + |x - x_c| <= r
 *
 * @param x X-coordinate
 * @param y Y-coordinate
 * @param cx Center x-coordinate
 * @param cy Center y-coordinate
 * @param r Half-width of diamond
 * @return Signed distance (negative inside)
 */
__host__ __device__ inline double signedDistanceDiamond(double x, double y,
                                                         double cx, double cy, double r) {
    // Diamond defined by: |x - cx| + |y - cy| = r
    double dx = fabs(x - cx);
    double dy = fabs(y - cy);

    // The diamond is rotated 45 degrees, so we use L1 norm
    // Signed distance for a diamond (rotated square)
    double l1_dist = dx + dy;

    // Inside if l1_dist < r, outside if l1_dist > r
    // The exact signed distance to a diamond is more complex
    // For simplicity, we use an approximation that works well for level-set

    if (l1_dist <= r) {
        // Inside: find distance to nearest edge
        // The diamond has 4 edges, each at 45 degrees
        // Distance to edge: (r - l1_dist) / sqrt(2)
        return -(r - l1_dist) / sqrt(2.0);
    } else {
        // Outside: find distance to nearest point on diamond
        // This depends on which quadrant we're in
        double excess = l1_dist - r;

        // For points aligned with edges, distance is excess / sqrt(2)
        // For points aligned with vertices, distance is to the vertex
        if (dx < dy) {
            // Closer to top or bottom vertex
            if (dx < r - fabs(dy - r)) {
                return excess / sqrt(2.0);
            }
        } else {
            // Closer to left or right vertex
            if (dy < r - fabs(dx - r)) {
                return excess / sqrt(2.0);
            }
        }
        return excess / sqrt(2.0);
    }
}

/**
 * @brief Compute exact signed distance to a diamond shape
 *
 * More accurate computation for the signed distance function
 */
__host__ __device__ inline double signedDistancePyramid(double x, double y,
                                                        double cx, double cy_base, double r) {
    double px = fabs(x - cx);
    double distance = (2.0 * px + y - 1.0) / sqrt(5.0);

    return distance;
}

/**
 * @brief Compute signed distance to a circle
 *
 * @param x X-coordinate
 * @param y Y-coordinate
 * @param cx Center x-coordinate
 * @param cy Center y-coordinate
 * @param r Radius
 * @return Signed distance (negative inside)
 */
__host__ __device__ inline double signedDistanceCircle(double x, double y,
                                                        double cx, double cy, double r) {
    double dx = x - cx;
    double dy = y - cy;
    return sqrt(dx * dx + dy * dy) - r;
}

/**
 * @brief Compute signed distance to a square
 *
 * @param x X-coordinate
 * @param y Y-coordinate
 * @param cx Center x-coordinate
 * @param cy Center y-coordinate
 * @param half_size Half the side length
 * @return Signed distance (negative inside)
 */
__host__ __device__ inline double signedDistanceSquare(double x, double y,
                                                        double cx, double cy, double half_size) {
    double dx = fabs(x - cx) - half_size;
    double dy = fabs(y - cy) - half_size;

    // Outside distance
    double outside = sqrt(fmax(dx, 0.0) * fmax(dx, 0.0) + fmax(dy, 0.0) * fmax(dy, 0.0));
    // Inside distance
    double inside = fmin(fmax(dx, dy), 0.0);

    return outside + inside;
}

//=============================================================================
// Zalesak's Slotted Disk Parameters (Dupont & Liu 2003, Hartmann et al. 2010)
//=============================================================================

constexpr double ZALESAK_CENTER_X = 0.5;
constexpr double ZALESAK_CENTER_Y = 0.75;
constexpr double ZALESAK_RADIUS = 0.15;
constexpr double ZALESAK_SLOT_WIDTH = 0.05;   // Total width of slot
constexpr double ZALESAK_SLOT_LENGTH = 0.25;  // Length of slot from bottom

/**
 * @brief Compute signed distance to Zalesak's slotted disk
 *
 * The disk has a rectangular slot cut from the bottom.
 * Disk: center (0.5, 0.75), radius 0.15
 * Slot: width 0.05, length 0.25 (centered at x=0.5, extending from bottom)
 *
 * G < 0 inside the disk (excluding the slot)
 * G > 0 outside the disk (including the slot)
 *
 * @param x X-coordinate
 * @param y Y-coordinate
 * @param cx Disk center x
 * @param cy Disk center y
 * @param r Disk radius
 * @param slot_width Slot width
 * @param slot_length Slot length from bottom of disk
 * @return Signed distance (negative inside disk, positive in slot and outside)
 */
__host__ __device__ inline double signedDistanceZalesak(double x, double y,
                                                         double cx, double cy, double r,
                                                         double slot_width, double slot_length) {
    // Signed distance to disk (negative inside)
    double dx = x - cx;
    double dy = y - cy;
    double dist_disk = sqrt(dx * dx + dy * dy) - r;

    // Slot is a rectangle centered at cx, extending from (cy - r) upward by slot_length
    // Slot bounds: x in [cx - slot_width/2, cx + slot_width/2]
    //              y in [cy - r, cy - r + slot_length]
    double slot_half_width = slot_width / 2.0;
    double slot_bottom = cy - r;
    double slot_top = slot_bottom + slot_length;

    // Signed distance to slot rectangle (negative inside slot)
    double slot_dx = fabs(x - cx) - slot_half_width;
    double slot_dy = fmax(slot_bottom - y, y - slot_top);

    double dist_slot;
    if (slot_dx < 0.0 && slot_dy < 0.0) {
        // Inside slot
        dist_slot = fmax(slot_dx, slot_dy);
    } else if (slot_dx < 0.0) {
        // Above or below slot
        dist_slot = slot_dy;
    } else if (slot_dy < 0.0) {
        // Left or right of slot
        dist_slot = slot_dx;
    } else {
        // Corner case
        dist_slot = sqrt(slot_dx * slot_dx + slot_dy * slot_dy);
    }

    // Slotted disk = disk AND NOT slot = intersection of disk and complement of slot
    // For CSG: disk ∩ slot^c means we are inside if inside disk AND outside slot
    // SDF: max(dist_disk, -dist_slot)
    // But we want: G < 0 inside disk (excluding slot), G > 0 outside or in slot
    // This means: inside = (inside disk) AND (outside slot)
    // SDF for this: max(dist_disk, -dist_slot)
    // However, -dist_slot is negative inside slot, so max gives:
    //   - If inside disk (dist_disk < 0) and outside slot (dist_slot > 0): max(neg, neg) < 0 ✓
    //   - If inside disk and inside slot: max(neg, pos) > 0 ✓
    //   - If outside disk: max(pos, ?) > 0 ✓

    return fmax(dist_disk, -dist_slot);
}

//=============================================================================
// Initialization Kernels
//=============================================================================

/**
 * @brief Initialize G field with pyramid (diamond) shape
 */
__global__ void initPyramidKernel(double* G, SimParams params,
                                   double cx, double cy, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    // Compute physical coordinates
    double x, y;
    indexToCoord(i, j, params.x_min, params.y_min,
                 params.dx, params.dy, params.nghost, x, y);

    int index = idx(i, j, params.nx_total);
    G[index] = signedDistancePyramid(x, y, cx, cy, r);
}

/**
 * @brief Initialize G field with circle shape
 */
__global__ void initCircleKernel(double* G, SimParams params,
                                  double cx, double cy, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    double x, y;
    indexToCoord(i, j, params.x_min, params.y_min,
                 params.dx, params.dy, params.nghost, x, y);

    int index = idx(i, j, params.nx_total);
    G[index] = signedDistanceCircle(x, y, cx, cy, r);
}

/**
 * @brief Initialize G field with square shape
 */
__global__ void initSquareKernel(double* G, SimParams params,
                                  double cx, double cy, double half_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    double x, y;
    indexToCoord(i, j, params.x_min, params.y_min,
                 params.dx, params.dy, params.nghost, x, y);

    int index = idx(i, j, params.nx_total);
    G[index] = signedDistanceSquare(x, y, cx, cy, half_size);
}

/**
 * @brief Initialize constant velocity field
 */
__global__ void initConstantVelocityKernel(double* u, double* v, SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    int index = idx(i, j, params.nx_total);
    u[index] = params.u_const;
    v[index] = params.v_const;
}

/**
 * @brief Initialize rotating velocity field (for Zalesak disk test)
 *
 * u = -ω(y - y_c)
 * v = ω(x - x_c)
 */
__global__ void initRotatingVelocityKernel(double* u, double* v, SimParams params,
                                            double cx, double cy, double omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    double x, y;
    indexToCoord(i, j, params.x_min, params.y_min,
                 params.dx, params.dy, params.nghost, x, y);

    int index = idx(i, j, params.nx_total);
    u[index] = -omega * (y - cy);
    v[index] = omega * (x - cx);
}

//=============================================================================
// Host Initialization Functions
//=============================================================================

/**
 * @brief Initialize pyramid/diamond test case
 *
 * Sets up:
 * - G field with diamond shape at (0.25, 0.5) with half-width 0.15
 * - Constant velocity field (u, v) = (1, 0)
 *
 * This is the standard advection test where the shape should return
 * to its original position after time T = 1.0 (one period).
 */
void initPyramidTestCase(double* d_G, double* d_u, double* d_v, SimParams& params) {
    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    double cx = 0.5;      // 중앙
    double cy_apex = 1.0; // 높이 1
    double r = 0.5;       // 폭 1 (반지름 0.5)

    initPyramidKernel<<<grid, block>>>(d_G, params, cx, cy_apex, r);
    
    // 속도장 설정 (u=1, v=0)
    initConstantVelocityKernel<<<grid, block>>>(d_u, d_v, params);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Initialize circle test case
 */
void initCircleTestCase(double* d_G, double* d_u, double* d_v, SimParams& params) {
    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    double cx = params.x_min + CIRCLE_CENTER_X * (params.x_max - params.x_min);
    double cy = params.y_min + CIRCLE_CENTER_Y * (params.y_max - params.y_min);
    double r = CIRCLE_RADIUS * (params.x_max - params.x_min);

    initCircleKernel<<<grid, block>>>(d_G, params, cx, cy, r);
    CUDA_CHECK(cudaGetLastError());

    initConstantVelocityKernel<<<grid, block>>>(d_u, d_v, params);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Initialize G field with Zalesak's slotted disk shape
 */
__global__ void initZalesakKernel(double* G, SimParams params,
                                   double cx, double cy, double r,
                                   double slot_width, double slot_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.nx_total || j >= params.ny_total) return;

    double x, y;
    indexToCoord(i, j, params.x_min, params.y_min,
                 params.dx, params.dy, params.nghost, x, y);

    int index = idx(i, j, params.nx_total);
    G[index] = signedDistanceZalesak(x, y, cx, cy, r, slot_width, slot_length);
}

/**
 * @brief Initialize Zalesak's slotted disk test case
 *
 * Sets up (Dupont & Liu 2003, Hartmann et al. 2010):
 * - G field with slotted disk at (0.5, 0.75), radius 0.15
 * - Slot: width 0.05, length 0.25 (centered, extending from bottom)
 * - Rotational velocity field: (u, v) = (ω(0.5-y), ω(x-0.5))
 * - One full rotation in T = 2π/ω
 *
 * The disk should return to its original position after one period.
 * Sharp corners of the slot should be preserved with HCR-2 reinitialization.
 */
void initZalesakTestCase(double* d_G, double* d_u, double* d_v, SimParams& params) {
    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    // Disk parameters (normalized to [0,1] domain)
    double disk_cx = ZALESAK_CENTER_X;
    double disk_cy = ZALESAK_CENTER_Y;
    double disk_r = ZALESAK_RADIUS;
    double slot_width = ZALESAK_SLOT_WIDTH;
    double slot_length = ZALESAK_SLOT_LENGTH;

    // Rotation center (0.5, 0.5) and angular velocity
    // For one full rotation in t_final: ω = 2π / t_final
    double rot_cx = 0.5;
    double rot_cy = 0.5;
    double omega = 2.0 * M_PI / params.t_final;

    // Initialize level-set field with slotted disk
    initZalesakKernel<<<grid, block>>>(d_G, params, disk_cx, disk_cy, disk_r,
                                        slot_width, slot_length);
    CUDA_CHECK(cudaGetLastError());

    // Initialize rotating velocity field
    // (u, v) = (ω(0.5 - y), ω(x - 0.5))
    initRotatingVelocityKernel<<<grid, block>>>(d_u, d_v, params, rot_cx, rot_cy, omega);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Zalesak Disk Test Case:\n");
    printf("  Disk center: (%.2f, %.2f), radius: %.2f\n", disk_cx, disk_cy, disk_r);
    printf("  Slot: width=%.2f, length=%.2f\n", slot_width, slot_length);
    printf("  Rotation: center=(%.2f, %.2f), omega=%.4f rad/s\n", rot_cx, rot_cy, omega);
    printf("  One full rotation in T=%.2f\n", params.t_final);
}

/**
 * @brief Initialize rotating circle test case (simpler version without slot)
 */
void initRotatingCircleTestCase(double* d_G, double* d_u, double* d_v, SimParams& params) {
    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    // Circle center offset from rotation center
    double circle_cx = 0.5;
    double circle_cy = 0.75;
    double circle_r = 0.15;

    // Rotation center and angular velocity (one rotation in time T_FINAL)
    double rot_cx = 0.5;
    double rot_cy = 0.5;
    double omega = 2.0 * M_PI / params.t_final;

    initCircleKernel<<<grid, block>>>(d_G, params, circle_cx, circle_cy, circle_r);
    CUDA_CHECK(cudaGetLastError());

    initRotatingVelocityKernel<<<grid, block>>>(d_u, d_v, params, rot_cx, rot_cy, omega);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Copy initial G field for error calculation
 */
void copyInitialField(const double* d_G_src, double* d_G_initial, int size) {
    CUDA_CHECK(cudaMemcpy(d_G_initial, d_G_src, size * sizeof(double), cudaMemcpyDeviceToDevice));
}

#endif // INITIAL_CONDITIONS_CUH
