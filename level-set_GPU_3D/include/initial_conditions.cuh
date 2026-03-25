/**
 * @file initial_conditions_3d.cuh
 * @brief Initial condition generators for 3D level-set field
 *
 * This module provides 3D initial conditions:
 * - Sphere for deformation test
 * - 3D deformation test velocity field
 */

#ifndef INITIAL_CONDITIONS_CUH
#define INITIAL_CONDITIONS_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "config.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================
// Sphere Shape Parameters for Deformation Test
//=============================================================================

constexpr double SPHERE_CENTER_X = 0.35;
constexpr double SPHERE_CENTER_Y = 0.35;
constexpr double SPHERE_CENTER_Z = 0.35;
constexpr double SPHERE_RADIUS = 0.15;

//=============================================================================
// Signed Distance Function Computations
//=============================================================================

/**
 * @brief Compute exact signed distance to a sphere
 *
 * @param x, y, z Coordinates
 * @param cx, cy, cz Center coordinates
 * @param r Radius
 * @return Signed distance (negative inside)
 */
__host__ __device__ inline double signedDistanceSphere(double x, double y, double z,
                                                        double cx, double cy, double cz, double r) {
    double dx = x - cx;
    double dy = y - cy;
    double dz = z - cz;
    return sqrt(dx * dx + dy * dy + dz * dz) - r;
}

//=============================================================================
// 3D Deformation Test Velocity Field
//=============================================================================

/**
 * @brief Compute time-dependent deformation test velocity field (3D vortex flow)
 *
 * u(x, y, z, t) = 2 sin²(πx) sin(2πy) sin(2πz) cos(πt/T)
 * v(x, y, z, t) = -sin(2πx) sin²(πy) sin(2πz) cos(πt/T)
 * w(x, y, z, t) = -sin(2πx) sin(2πy) sin²(πz) cos(πt/T)
 *
 * This is a divergence-free, time-reversible flow field that stretches
 * the sphere into a thin film at t=T/2, then returns it to the original
 * shape at t=T.
 *
 * @param x, y, z Coordinates
 * @param t Current time
 * @param T Period (total simulation time)
 * @param u, v, w Output velocity components
 */
__host__ __device__ inline void computeDeformationVelocity(double x, double y, double z,
                                                            double t, double T,
                                                            double& u, double& v, double& w) {
    double time_factor = cos(M_PI * t / T);

    double sin_pix = sin(M_PI * x);
    double sin_2pix = sin(2.0 * M_PI * x);
    double sin_piy = sin(M_PI * y);
    double sin_2piy = sin(2.0 * M_PI * y);
    double sin_piz = sin(M_PI * z);
    double sin_2piz = sin(2.0 * M_PI * z);

    // u = 2.0 * sin_pix * sin_pix * sin_2piy * sin_2piz;
    // v = -sin_2pix * sin_piy * sin_piy * sin_2piz;
    // w = -sin_2pix * sin_2piy * sin_piz * sin_piz;

    u = 2.0 * sin_pix * sin_pix * sin_2piy * sin_2piz * time_factor;
    v = -sin_2pix * sin_piy * sin_piy * sin_2piz * time_factor;
    w = -sin_2pix * sin_2piy * sin_piz * sin_piz * time_factor;
}

//=============================================================================
// Initialization Kernels
//=============================================================================

/**
 * @brief Initialize G field with sphere shape
 */
__global__ void initSphereKernel(double* G, SimParams params,
                                  double cx, double cy, double cz, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= params.nx_total || j >= params.ny_total || k >= params.nz_total) return;

    // Compute physical coordinates
    double x, y, z;
    indexToCoord(i, j, k, params.x_min, params.y_min, params.z_min,
                 params.dx, params.dy, params.dz, params.nghost, x, y, z);

    int index = idx(i, j, k, params.nx_total, params.ny_total);
    G[index] = signedDistanceSphere(x, y, z, cx, cy, cz, r);
}

/**
 * @brief Initialize deformation test velocity field (time-dependent)
 *
 * This kernel computes the velocity field at the current time t.
 * The velocity field should be updated at each time step.
 */
__global__ void initDeformationVelocityKernel(double* u, double* v, double* w,
                                               SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= params.nx_total || j >= params.ny_total || k >= params.nz_total) return;

    // Compute physical coordinates
    double x, y, z;
    indexToCoord(i, j, k, params.x_min, params.y_min, params.z_min,
                 params.dx, params.dy, params.dz, params.nghost, x, y, z);

    int index = idx(i, j, k, params.nx_total, params.ny_total);

    // Compute velocity components at current time
    double u_val, v_val, w_val;
    computeDeformationVelocity(x, y, z, params.current_time, params.t_final,
                               u_val, v_val, w_val);

    u[index] = u_val;
    v[index] = v_val;
    w[index] = w_val;
}

/**
 * @brief Initialize constant velocity field (for testing)
 */
__global__ void initConstantVelocityKernel(double* u, double* v, double* w, SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= params.nx_total || j >= params.ny_total || k >= params.nz_total) return;

    int index = idx(i, j, k, params.nx_total, params.ny_total);
    u[index] = params.u_const;
    v[index] = params.v_const;
    w[index] = params.w_const;
}

//=============================================================================
// Host Initialization Functions
//=============================================================================

/**
 * @brief Initialize sphere test case for 3D deformation test
 *
 * Sets up:
 * - G field with sphere at (0.35, 0.35, 0.35) with radius 0.15
 * - Time-dependent deformation velocity field
 *
 * The sphere should be stretched into a thin film at t=T/2 and return
 * to its original shape at t=T.
 */
void initSphereDeformationTest(double* d_G, double* d_u, double* d_v, double* d_w, SimParams& params) {
    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    double cx = SPHERE_CENTER_X;
    double cy = SPHERE_CENTER_Y;
    double cz = SPHERE_CENTER_Z;
    double r = SPHERE_RADIUS;

    // Initialize sphere level-set field
    initSphereKernel<<<grid, block>>>(d_G, params, cx, cy, cz, r);
    CUDA_CHECK(cudaGetLastError());

    // Initialize velocity field at t=0
    params.current_time = 0.0;
    initDeformationVelocityKernel<<<grid, block>>>(d_u, d_v, d_w, params);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Initialized Sphere Deformation Test:\n");
    printf("  Sphere center: (%.2f, %.2f, %.2f)\n", cx, cy, cz);
    printf("  Sphere radius: %.2f\n", r);
    printf("  Deformation period: T=%.2f\n", params.t_final);
    printf("  At t=T/2: maximum deformation\n");
    printf("  At t=T: return to original shape\n");
}

/**
 * @brief Update velocity field for time-dependent deformation test
 *
 * This should be called at the beginning of each time step to update
 * the velocity field to the current time.
 */
void updateDeformationVelocity(double* d_u, double* d_v, double* d_w, SimParams& params) {
    dim3 grid = getGridDim();
    dim3 block = getBlockDim();

    initDeformationVelocityKernel<<<grid, block>>>(d_u, d_v, d_w, params);
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
