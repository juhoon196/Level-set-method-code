#include "field_cuda.cuh"
#include <cstring>

// ============================================================
// Field2DGPU implementation
// ============================================================

Field2DGPU::Field2DGPU() : d_data(nullptr), h_data(nullptr) {}

Field2DGPU::~Field2DGPU() {
    free();
}

void Field2DGPU::allocate() {
    size_t size = NX_TOTAL * NY_TOTAL * sizeof(double);
    h_data = new double[NX_TOTAL * NY_TOTAL];
    CUDA_CHECK(cudaMalloc(&d_data, size));
    memset(h_data, 0, size);
}

void Field2DGPU::free() {
    if (h_data) {
        delete[] h_data;
        h_data = nullptr;
    }
    if (d_data) {
        cudaFree(d_data);
        d_data = nullptr;
    }
}

void Field2DGPU::copyToDevice() {
    size_t size = NX_TOTAL * NY_TOTAL * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
}

void Field2DGPU::copyToHost() {
    size_t size = NX_TOTAL * NY_TOTAL * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
}

void Field2DGPU::fill(double value) {
    for (int idx = 0; idx < NX_TOTAL * NY_TOTAL; ++idx) {
        h_data[idx] = value;
    }
}

// ============================================================
// Periodic boundary condition kernel
// ============================================================

__global__ void kernel_apply_periodic(double* G) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Left-right periodic (for each row)
    if (tid < NY_TOTAL) {
        int j = tid;
        for (int k = 0; k < GHOST; ++k) {
            G[idx2d(j, k)] = G[idx2d(j, NX + k)];
            G[idx2d(j, NX + GHOST + k)] = G[idx2d(j, GHOST + k)];
        }
    }

    // Top-bottom periodic (for each column)
    if (tid < NX_TOTAL) {
        int i = tid;
        for (int k = 0; k < GHOST; ++k) {
            G[idx2d(k, i)] = G[idx2d(NY + k, i)];
            G[idx2d(NY + GHOST + k, i)] = G[idx2d(GHOST + k, i)];
        }
    }
}

void apply_periodic_gpu(double* d_G) {
    int max_dim = (NX_TOTAL > NY_TOTAL) ? NX_TOTAL : NY_TOTAL;
    int blocks = (max_dim + 255) / 256;
    kernel_apply_periodic<<<blocks, 256>>>(d_G);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Anchored boundary condition kernel
// ============================================================

__global__ void kernel_apply_anchored_bc(double* G, const double* G_initial) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bottom boundary: anchored + linear extrapolation
    if (tid < NX_TOTAL) {
        int i = tid;
        G[idx2d(GHOST, i)] = G_initial[idx2d(GHOST, i)];  // anchoring at j=3
        G[idx2d(2, i)] = 2.0 * G[idx2d(3, i)] - G[idx2d(4, i)];
        G[idx2d(1, i)] = 2.0 * G[idx2d(2, i)] - G[idx2d(3, i)];
        G[idx2d(0, i)] = 2.0 * G[idx2d(1, i)] - G[idx2d(2, i)];
    }

    // Top boundary: linear extrapolation
    if (tid < NX_TOTAL) {
        int i = tid;
        G[idx2d(NY + 2, i)] = 2.0 * G[idx2d(NY + 1, i)] - G[idx2d(NY, i)];
        G[idx2d(NY + 3, i)] = 2.0 * G[idx2d(NY + 2, i)] - G[idx2d(NY + 1, i)];
        G[idx2d(NY + 4, i)] = 2.0 * G[idx2d(NY + 3, i)] - G[idx2d(NY + 2, i)];
        G[idx2d(NY + 5, i)] = 2.0 * G[idx2d(NY + 4, i)] - G[idx2d(NY + 3, i)];
    }

    // Left/Right boundaries: linear extrapolation
    if (tid < NY_TOTAL) {
        int j = tid;
        // Left
        G[idx2d(j, 2)] = 2.0 * G[idx2d(j, 3)] - G[idx2d(j, 4)];
        G[idx2d(j, 1)] = 2.0 * G[idx2d(j, 2)] - G[idx2d(j, 3)];
        G[idx2d(j, 0)] = 2.0 * G[idx2d(j, 1)] - G[idx2d(j, 2)];
        // Right
        G[idx2d(j, NX + 3)] = 2.0 * G[idx2d(j, NX + 2)] - G[idx2d(j, NX + 1)];
        G[idx2d(j, NX + 4)] = 2.0 * G[idx2d(j, NX + 3)] - G[idx2d(j, NX + 2)];
        G[idx2d(j, NX + 5)] = 2.0 * G[idx2d(j, NX + 4)] - G[idx2d(j, NX + 3)];
    }
}

void apply_anchored_bc_gpu(double* d_G, const double* d_G_initial) {
    int max_dim = (NX_TOTAL > NY_TOTAL) ? NX_TOTAL : NY_TOTAL;
    int blocks = (max_dim + 255) / 256;
    kernel_apply_anchored_bc<<<blocks, 256>>>(d_G, d_G_initial);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Copy field on device
// ============================================================

void copy_field_gpu(double* d_dst, const double* d_src) {
    size_t size = NX_TOTAL * NY_TOTAL * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
}
