/**
 * @file boundary.cuh
 * @brief Boundary conditions with MPI halo exchange (MPI + CUDA)
 *
 * X, Z directions: local periodic BC (same as single-GPU).
 * Y direction:     MPI halo exchange between neighbouring ranks.
 *                  Periodic wrap-around is handled by neighbour assignment.
 */

#ifndef BOUNDARY_CUH
#define BOUNDARY_CUH

#include <mpi.h>
#include <cuda_runtime.h>
#include "config.cuh"

//=============================================================================
// Halo Buffer Management
//=============================================================================

struct HaloBuffers {
    double *d_send_below, *d_send_above;   // device pack buffers
    double *d_recv_below, *d_recv_above;   // device unpack buffers
    double *h_send_below, *h_send_above;   // host staging (pinned)
    double *h_recv_below, *h_recv_above;
    int halo_size;   // NGHOST * nx_total * nz_total (doubles)
};

inline void allocHaloBuffers(HaloBuffers& hb, const SimParams& p) {
    hb.halo_size = p.nghost * p.nx_total * p.nz_total;
    size_t bytes = hb.halo_size * sizeof(double);

    CUDA_CHECK(cudaMalloc(&hb.d_send_below, bytes));
    CUDA_CHECK(cudaMalloc(&hb.d_send_above, bytes));
    CUDA_CHECK(cudaMalloc(&hb.d_recv_below, bytes));
    CUDA_CHECK(cudaMalloc(&hb.d_recv_above, bytes));

    CUDA_CHECK(cudaMallocHost(&hb.h_send_below, bytes));
    CUDA_CHECK(cudaMallocHost(&hb.h_send_above, bytes));
    CUDA_CHECK(cudaMallocHost(&hb.h_recv_below, bytes));
    CUDA_CHECK(cudaMallocHost(&hb.h_recv_above, bytes));
}

inline void freeHaloBuffers(HaloBuffers& hb) {
    cudaFree(hb.d_send_below);  cudaFree(hb.d_send_above);
    cudaFree(hb.d_recv_below);  cudaFree(hb.d_recv_above);
    cudaFreeHost(hb.h_send_below); cudaFreeHost(hb.h_send_above);
    cudaFreeHost(hb.h_recv_below); cudaFreeHost(hb.h_recv_above);
}

//=============================================================================
// Pack / Unpack Kernels for Y-Halo
//=============================================================================

/**
 * Pack NGHOST y-layers starting at j = j_start into a contiguous buffer.
 * Buffer layout: buf[g * nz_total * nx_total + k * nx_total + i]
 */
__global__ void packYHalo(const double* __restrict__ G, double* __restrict__ buf,
                           int j_start, int nghost,
                           int nx_total, int ny_total, int nz_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx_total || k >= nz_total) return;

    for (int g = 0; g < nghost; g++) {
        buf[g * nz_total * nx_total + k * nx_total + i] =
            G[(k * ny_total + (j_start + g)) * nx_total + i];
    }
}

__global__ void unpackYHalo(double* __restrict__ G, const double* __restrict__ buf,
                             int j_start, int nghost,
                             int nx_total, int ny_total, int nz_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx_total || k >= nz_total) return;

    for (int g = 0; g < nghost; g++) {
        G[(k * ny_total + (j_start + g)) * nx_total + i] =
            buf[g * nz_total * nx_total + k * nx_total + i];
    }
}

//=============================================================================
// Local Periodic BC Kernels (X and Z — unchanged from single-GPU)
//=============================================================================

__global__ void applyPeriodicBC_X_3D(double* G, int nx, int nghost,
                                      int nx_total, int ny_total, int nz_total) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny_total || k >= nz_total) return;

    for (int g = 0; g < nghost; g++) {
        // Left ghost ← right interior
        G[idx(g, j, k, nx_total, ny_total)] =
            G[idx(nx + g, j, k, nx_total, ny_total)];
        // Right ghost ← left interior
        G[idx(nx + nghost + g, j, k, nx_total, ny_total)] =
            G[idx(nghost + g, j, k, nx_total, ny_total)];
    }
}

__global__ void applyPeriodicBC_Z_3D(double* G, int nz, int nghost,
                                      int nx_total, int ny_total, int nz_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx_total || j >= ny_total) return;

    for (int g = 0; g < nghost; g++) {
        // Back ghost ← front interior
        G[idx(i, j, g, nx_total, ny_total)] =
            G[idx(i, j, nz + g, nx_total, ny_total)];
        // Front ghost ← back interior
        G[idx(i, j, nz + nghost + g, nx_total, ny_total)] =
            G[idx(i, j, nghost + g, nx_total, ny_total)];
    }
}

//=============================================================================
// Y-Halo Exchange (MPI + CUDA)
//=============================================================================

/**
 * Exchange NGHOST y-layers between neighbouring ranks.
 *
 *   rank's local array layout in j:
 *     [0 .. NGHOST-1]                         bottom ghost
 *     [NGHOST .. NGHOST+local_ny-1]           interior
 *     [NGHOST+local_ny .. NGHOST+local_ny+NGHOST-1]  top ghost
 *
 *   Send bottom interior  → neighbor_below  (fills their top ghost)
 *   Send top    interior  → neighbor_above  (fills their bottom ghost)
 */
inline void exchangeYHalo(double* d_G, const SimParams& p, HaloBuffers& hb) {
    int ng = p.nghost;
    int nx_t = p.nx_total, ny_t = p.ny_total, nz_t = p.nz_total;
    size_t bytes = hb.halo_size * sizeof(double);

    int threads = 16;
    dim3 block2d(threads, threads);
    dim3 grid2d((nx_t + threads - 1) / threads,
                (nz_t + threads - 1) / threads);

    // 1. Pack on GPU
    //    bottom interior: j_start = nghost
    packYHalo<<<grid2d, block2d>>>(d_G, hb.d_send_below,
                                    ng, ng, nx_t, ny_t, nz_t);
    //    top interior: j_start = nghost + local_ny - nghost = local_ny
    packYHalo<<<grid2d, block2d>>>(d_G, hb.d_send_above,
                                    p.local_ny, ng, nx_t, ny_t, nz_t);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. Device → Host
    CUDA_CHECK(cudaMemcpy(hb.h_send_below, hb.d_send_below, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.h_send_above, hb.d_send_above, bytes, cudaMemcpyDeviceToHost));

    // 3. MPI exchange (non-blocking for overlap potential)
    MPI_Request reqs[4];
    // Send bottom interior to below, recv bottom ghost from below
    MPI_CHECK(MPI_Isend(hb.h_send_below, hb.halo_size, MPI_DOUBLE,
                         p.neighbor_below, 0, MPI_COMM_WORLD, &reqs[0]));
    MPI_CHECK(MPI_Irecv(hb.h_recv_below, hb.halo_size, MPI_DOUBLE,
                         p.neighbor_below, 1, MPI_COMM_WORLD, &reqs[1]));
    // Send top interior to above, recv top ghost from above
    MPI_CHECK(MPI_Isend(hb.h_send_above, hb.halo_size, MPI_DOUBLE,
                         p.neighbor_above, 1, MPI_COMM_WORLD, &reqs[2]));
    MPI_CHECK(MPI_Irecv(hb.h_recv_above, hb.halo_size, MPI_DOUBLE,
                         p.neighbor_above, 0, MPI_COMM_WORLD, &reqs[3]));
    MPI_CHECK(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE));

    // 4. Host → Device
    CUDA_CHECK(cudaMemcpy(hb.d_recv_below, hb.h_recv_below, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hb.d_recv_above, hb.h_recv_above, bytes, cudaMemcpyHostToDevice));

    // 5. Unpack on GPU
    //    bottom ghost: j_start = 0
    unpackYHalo<<<grid2d, block2d>>>(d_G, hb.d_recv_below,
                                      0, ng, nx_t, ny_t, nz_t);
    //    top ghost: j_start = nghost + local_ny
    unpackYHalo<<<grid2d, block2d>>>(d_G, hb.d_recv_above,
                                      ng + p.local_ny, ng, nx_t, ny_t, nz_t);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Enum to select the optimization mode
enum HaloOptMode {
    ORIGINAL = 0,         // Original (Synchronous copy + Host communication)
    CUDA_AWARE_MPI = 1,   // Optimization 1 (Skip copy + Direct GPU communication)
    ASYNC_STREAMS = 2     // Optimization 2 (Asynchronous copy + Overlap)
};

inline void exchangeYHalo_Switchable(double* d_G, const SimParams& p, HaloBuffers& hb, 
                                     HaloOptMode mode, cudaStream_t stream_below = 0, cudaStream_t stream_above = 0) {
    int ng = p.nghost;
    int nx_t = p.nx_total, ny_t = p.ny_total, nz_t = p.nz_total;
    size_t bytes = hb.halo_size * sizeof(double);

    int threads = 16;
    dim3 block2d(threads, threads);
    dim3 grid2d((nx_t + threads - 1) / threads,
                (nz_t + threads - 1) / threads);

    MPI_Request reqs[4];

    switch (mode) {
        case ORIGINAL:
            // [Original Method]: Pack -> Full Sync Wait -> D2H -> MPI -> H2D -> Unpack -> Full Sync Wait
            packYHalo<<<grid2d, block2d>>>(d_G, hb.d_send_below, ng, ng, nx_t, ny_t, nz_t);
            packYHalo<<<grid2d, block2d>>>(d_G, hb.d_send_above, p.local_ny, ng, nx_t, ny_t, nz_t);
            CUDA_CHECK(cudaDeviceSynchronize()); // Bottleneck 1

            CUDA_CHECK(cudaMemcpy(hb.h_send_below, hb.d_send_below, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hb.h_send_above, hb.d_send_above, bytes, cudaMemcpyDeviceToHost));

            MPI_CHECK(MPI_Isend(hb.h_send_below, hb.halo_size, MPI_DOUBLE, p.neighbor_below, 0, MPI_COMM_WORLD, &reqs[0]));
            MPI_CHECK(MPI_Irecv(hb.h_recv_below, hb.halo_size, MPI_DOUBLE, p.neighbor_below, 1, MPI_COMM_WORLD, &reqs[1]));
            MPI_CHECK(MPI_Isend(hb.h_send_above, hb.halo_size, MPI_DOUBLE, p.neighbor_above, 1, MPI_COMM_WORLD, &reqs[2]));
            MPI_CHECK(MPI_Irecv(hb.h_recv_above, hb.halo_size, MPI_DOUBLE, p.neighbor_above, 0, MPI_COMM_WORLD, &reqs[3]));
            MPI_CHECK(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE));

            CUDA_CHECK(cudaMemcpy(hb.d_recv_below, hb.h_recv_below, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(hb.d_recv_above, hb.h_recv_above, bytes, cudaMemcpyHostToDevice));

            unpackYHalo<<<grid2d, block2d>>>(d_G, hb.d_recv_below, 0, ng, nx_t, ny_t, nz_t);
            unpackYHalo<<<grid2d, block2d>>>(d_G, hb.d_recv_above, ng + p.local_ny, ng, nx_t, ny_t, nz_t);
            CUDA_CHECK(cudaDeviceSynchronize()); // Bottleneck 2
            break;

        case CUDA_AWARE_MPI:
            // [Optimization 1]: Completely remove cudaMemcpy and use GPU pointers (d_send, d_recv) directly for MPI
            packYHalo<<<grid2d, block2d>>>(d_G, hb.d_send_below, ng, ng, nx_t, ny_t, nz_t);
            packYHalo<<<grid2d, block2d>>>(d_G, hb.d_send_above, p.local_ny, ng, nx_t, ny_t, nz_t);
            CUDA_CHECK(cudaDeviceSynchronize()); 

            // Note: Using hb.d_send instead of hb.h_send!
            MPI_CHECK(MPI_Isend(hb.d_send_below, hb.halo_size, MPI_DOUBLE, p.neighbor_below, 0, MPI_COMM_WORLD, &reqs[0]));
            MPI_CHECK(MPI_Irecv(hb.d_recv_below, hb.halo_size, MPI_DOUBLE, p.neighbor_below, 1, MPI_COMM_WORLD, &reqs[1]));
            MPI_CHECK(MPI_Isend(hb.d_send_above, hb.halo_size, MPI_DOUBLE, p.neighbor_above, 1, MPI_COMM_WORLD, &reqs[2]));
            MPI_CHECK(MPI_Irecv(hb.d_recv_above, hb.halo_size, MPI_DOUBLE, p.neighbor_above, 0, MPI_COMM_WORLD, &reqs[3]));
            MPI_CHECK(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE));

            unpackYHalo<<<grid2d, block2d>>>(d_G, hb.d_recv_below, 0, ng, nx_t, ny_t, nz_t);
            unpackYHalo<<<grid2d, block2d>>>(d_G, hb.d_recv_above, ng + p.local_ny, ng, nx_t, ny_t, nz_t);
            CUDA_CHECK(cudaDeviceSynchronize());
            break;

        case ASYNC_STREAMS:
            // [Optimization 2]: Separate top/bottom tasks into different streams to overlap packing and memory copies
            // Step 1: Stream-separated packing and async copy (D2H)
            packYHalo<<<grid2d, block2d, 0, stream_below>>>(d_G, hb.d_send_below, ng, ng, nx_t, ny_t, nz_t);
            CUDA_CHECK(cudaMemcpyAsync(hb.h_send_below, hb.d_send_below, bytes, cudaMemcpyDeviceToHost, stream_below));

            packYHalo<<<grid2d, block2d, 0, stream_above>>>(d_G, hb.d_send_above, p.local_ny, ng, nx_t, ny_t, nz_t);
            CUDA_CHECK(cudaMemcpyAsync(hb.h_send_above, hb.d_send_above, bytes, cudaMemcpyDeviceToHost, stream_above));

            // Wait for copies to finish before MPI communication
            CUDA_CHECK(cudaStreamSynchronize(stream_below));
            CUDA_CHECK(cudaStreamSynchronize(stream_above));

            // Step 2: MPI Communication (Host memory)
            MPI_CHECK(MPI_Isend(hb.h_send_below, hb.halo_size, MPI_DOUBLE, p.neighbor_below, 0, MPI_COMM_WORLD, &reqs[0]));
            MPI_CHECK(MPI_Irecv(hb.h_recv_below, hb.halo_size, MPI_DOUBLE, p.neighbor_below, 1, MPI_COMM_WORLD, &reqs[1]));
            MPI_CHECK(MPI_Isend(hb.h_send_above, hb.halo_size, MPI_DOUBLE, p.neighbor_above, 1, MPI_COMM_WORLD, &reqs[2]));
            MPI_CHECK(MPI_Irecv(hb.h_recv_above, hb.halo_size, MPI_DOUBLE, p.neighbor_above, 0, MPI_COMM_WORLD, &reqs[3]));
            MPI_CHECK(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE));

            // Step 3: Stream-separated async copy (H2D) and unpacking
            CUDA_CHECK(cudaMemcpyAsync(hb.d_recv_below, hb.h_recv_below, bytes, cudaMemcpyHostToDevice, stream_below));
            unpackYHalo<<<grid2d, block2d, 0, stream_below>>>(d_G, hb.d_recv_below, 0, ng, nx_t, ny_t, nz_t);

            CUDA_CHECK(cudaMemcpyAsync(hb.d_recv_above, hb.h_recv_above, bytes, cudaMemcpyHostToDevice, stream_above));
            unpackYHalo<<<grid2d, block2d, 0, stream_above>>>(d_G, hb.d_recv_above, ng + p.local_ny, ng, nx_t, ny_t, nz_t);

            // Final wait
            CUDA_CHECK(cudaStreamSynchronize(stream_below));
            CUDA_CHECK(cudaStreamSynchronize(stream_above));
            break;
    }
}

//=============================================================================
// Combined Boundary Conditions
//=============================================================================

inline void applyBoundaryConditions(double* d_G, SimParams& p, HaloBuffers& hb) {
    int threads = 16;
    dim3 block2d(threads, threads);

    // X periodic (local)
    dim3 gridX((p.ny_total + threads - 1) / threads,
               (p.nz_total + threads - 1) / threads);
    applyPeriodicBC_X_3D<<<gridX, block2d>>>(d_G, p.nx, p.nghost,
                                              p.nx_total, p.ny_total, p.nz_total);

    // Z periodic (local)
    dim3 gridZ((p.nx_total + threads - 1) / threads,
               (p.ny_total + threads - 1) / threads);
    applyPeriodicBC_Z_3D<<<gridZ, block2d>>>(d_G, p.nz, p.nghost,
                                              p.nx_total, p.ny_total, p.nz_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Y halo exchange (MPI)
    //exchangeYHalo(d_G, p, hb);
    //  ORIGINAL = 0,         // Original (Synchronous copy + Host communication)
    //  CUDA_AWARE_MPI = 1,   // Optimization 1 (Skip copy + Direct GPU communication)
    //  ASYNC_STREAMS = 2     // Optimization 2 (Asynchronous copy + Overlap)

    //exchangeYHalo_Switchable(d_G, p, hb, ORIGINAL);
    exchangeYHalo_Switchable(d_G, p, hb, CUDA_AWARE_MPI);
    //exchangeYHalo_Switchable(d_G, p, hb, ASYNC_STREAMS);
}

inline void applyVelocityBoundaryConditions(double* d_u, double* d_v, double* d_w,
                                              SimParams& p, HaloBuffers& hb) {
    applyBoundaryConditions(d_u, p, hb);
    applyBoundaryConditions(d_v, p, hb);
    applyBoundaryConditions(d_w, p, hb);
}

#endif // BOUNDARY_CUH
