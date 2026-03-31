/**
 * @file initial_conditions.cuh
 * @brief Initial conditions for 3D level-set field (MPI + CUDA)
 *
 * Uses y_start offset so each rank initialises its own slab correctly.
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
// Shape Parameters
//=============================================================================

constexpr double SPHERE_CENTER_X = 0.35;
constexpr double SPHERE_CENTER_Y = 0.35;
constexpr double SPHERE_CENTER_Z = 0.35;
constexpr double SPHERE_RADIUS   = 0.15;

//=============================================================================
// Device Functions
//=============================================================================

__host__ __device__ inline double signedDistanceSphere(double x, double y, double z,
                                                        double cx, double cy, double cz, double r) {
    double dx = x - cx, dy = y - cy, dz = z - cz;
    return sqrt(dx*dx + dy*dy + dz*dz) - r;
}

__host__ __device__ inline void computeDeformationVelocity(double x, double y, double z,
                                                            double t, double T,
                                                            double& u, double& v, double& w) {
    double tf = cos(M_PI * t / T);
    double spx = sin(M_PI * x), s2px = sin(2.0 * M_PI * x);
    double spy = sin(M_PI * y), s2py = sin(2.0 * M_PI * y);
    double spz = sin(M_PI * z), s2pz = sin(2.0 * M_PI * z);

    u =  2.0 * spx * spx * s2py * s2pz * tf;
    v = -s2px * spy * spy * s2pz * tf;
    w = -s2px * s2py * spz * spz * tf;
}

//=============================================================================
// Kernels (use y_start for global coordinate)
//=============================================================================

__global__ void initSphereKernel(double* G, SimParams params,
                                  double cx, double cy, double cz, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= params.nx_total || j >= params.ny_total || k >= params.nz_total) return;

    double x, y, z;
    indexToCoord(i, j, k, params.x_min, params.y_min, params.z_min,
                 params.dx, params.dy, params.dz, params.nghost,
                 params.y_start, x, y, z);

    G[idx(i, j, k, params.nx_total, params.ny_total)] =
        signedDistanceSphere(x, y, z, cx, cy, cz, r);
}

__global__ void initDeformationVelocityKernel(double* u, double* v, double* w,
                                               SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= params.nx_total || j >= params.ny_total || k >= params.nz_total) return;

    double x, y, z;
    indexToCoord(i, j, k, params.x_min, params.y_min, params.z_min,
                 params.dx, params.dy, params.dz, params.nghost,
                 params.y_start, x, y, z);

    int index = idx(i, j, k, params.nx_total, params.ny_total);
    double uv, vv, wv;
    computeDeformationVelocity(x, y, z, params.current_time, params.t_final, uv, vv, wv);
    u[index] = uv;  v[index] = vv;  w[index] = wv;
}

//=============================================================================
// Host Wrappers
//=============================================================================

void initSphereDeformationTest(double* d_G, double* d_u, double* d_v, double* d_w,
                                SimParams& params) {
    dim3 grid = getGridDim(params);
    dim3 block = getBlockDim();

    double cx = SPHERE_CENTER_X, cy = SPHERE_CENTER_Y, cz = SPHERE_CENTER_Z;
    double r  = SPHERE_RADIUS;

    initSphereKernel<<<grid, block>>>(d_G, params, cx, cy, cz, r);
    CUDA_CHECK(cudaGetLastError());

    params.current_time = 0.0;
    initDeformationVelocityKernel<<<grid, block>>>(d_u, d_v, d_w, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (params.rank == 0) {
        printf("Initialized Sphere Deformation Test:\n");
        printf("  Sphere center: (%.2f, %.2f, %.2f)\n", cx, cy, cz);
        printf("  Sphere radius: %.2f\n", r);
        printf("  Deformation period: T=%.2f\n", params.t_final);
    }
}

void updateDeformationVelocity(double* d_u, double* d_v, double* d_w,
                                SimParams& params) {
    dim3 grid = getGridDim(params);
    dim3 block = getBlockDim();
    initDeformationVelocityKernel<<<grid, block>>>(d_u, d_v, d_w, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void copyInitialField(const double* d_src, double* d_dst, int size) {
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size * sizeof(double), cudaMemcpyDeviceToDevice));
}

#endif // INITIAL_CONDITIONS_CUH
