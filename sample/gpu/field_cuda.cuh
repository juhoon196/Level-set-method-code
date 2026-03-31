#pragma once
#include "config_cuda.cuh"
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 2D indexing helper (row-major)
__host__ __device__ inline int idx2d(int j, int i) {
    return j * NX_TOTAL + i;
}

// Field2D class for GPU
class Field2DGPU {
public:
    double* d_data;      // Device pointer
    double* h_data;      // Host pointer (for I/O)

    Field2DGPU();
    ~Field2DGPU();

    void allocate();
    void free();
    void copyToDevice();
    void copyToHost();
    void fill(double value);

    // Access host data for initialization/output
    double& at(int j, int i) { return h_data[idx2d(j, i)]; }
    const double& at(int j, int i) const { return h_data[idx2d(j, i)]; }
};

// Boundary condition kernels
__global__ void kernel_apply_periodic(double* G);
__global__ void kernel_apply_anchored_bc(double* G, const double* G_initial);

// Wrapper functions
void apply_periodic_gpu(double* d_G);
void apply_anchored_bc_gpu(double* d_G, const double* d_G_initial);

// Utility: copy field on device
void copy_field_gpu(double* d_dst, const double* d_src);
