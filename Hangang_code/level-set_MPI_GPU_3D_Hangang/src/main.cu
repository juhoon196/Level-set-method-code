/**
 * @file main.cu
 * @brief MPI + CUDA multi-GPU 3D Level-Set Deformation Test
 *
 * Each MPI rank owns one GPU and a Y-slab of the domain.
 * Y-halo exchange is done via MPI (pack on GPU → D2H → MPI → H2D → unpack).
 * X and Z periodic BCs are handled locally on each GPU.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>

#include "../include/config.cuh"
#include "../include/weno5.cuh"
#include "../include/initial_conditions.cuh"
#include "../include/rk3.cuh"
#include "../include/boundary.cuh"

//=============================================================================
// I/O Helpers (rank 0 gathers and writes)
//=============================================================================

/**
 * Pack local interior Y-rows (excluding Y ghosts) into a contiguous host buffer.
 * Layout: k * nx_total * local_ny + j * nx_total + i
 */
static void packLocalInterior(const double* h_local, double* h_interior,
                                const SimParams& p) {
    int nx_t = p.nx_total, ny_t = p.ny_total, nz_t = p.nz_total;
    int ng = p.nghost, lny = p.local_ny;
    int pos = 0;
    for (int k = 0; k < nz_t; k++)
        for (int j = 0; j < lny; j++)
            for (int i = 0; i < nx_t; i++)
                h_interior[pos++] = h_local[k * nx_t * ny_t + (j + ng) * nx_t + i];
}

/**
 * Rank 0: reconstruct full array (with Y ghost cells) from gathered interior data.
 */
static void reconstructFullField(const double* global_interior,
                                   double* full_G,
                                   const SimParams& p,
                                   const int* recv_counts,
                                   const int* displs) {
    int nx_t = p.nx_total, nz_t = p.nz_total;
    int ng = p.nghost;
    int gny = p.global_ny;
    int ny_total_g = gny + 2 * ng;
    int full_size = nx_t * ny_total_g * nz_t;

    // Zero-init (ghost cells default to 0, will be filled below)
    memset(full_G, 0, full_size * sizeof(double));

    // Place each rank's interior rows at the correct global position
    for (int r = 0; r < p.num_procs; r++) {
        int base_rows = gny / p.num_procs;
        int extra     = gny % p.num_procs;
        int lny = (r < extra) ? base_rows + 1 : base_rows;
        int ys  = (r < extra) ? r * (base_rows + 1)
                              : extra * (base_rows + 1) + (r - extra) * base_rows;
        int offset = displs[r];

        for (int k = 0; k < nz_t; k++)
            for (int j = 0; j < lny; j++)
                for (int i = 0; i < nx_t; i++) {
                    int src = offset + k * nx_t * lny + j * nx_t + i;
                    int dst = k * nx_t * ny_total_g + (ys + j + ng) * nx_t + i;
                    full_G[dst] = global_interior[src];
                }
    }

    // Fill Y ghost cells (periodic: wrap around)
    for (int k = 0; k < nz_t; k++)
        for (int i = 0; i < nx_t; i++)
            for (int g = 0; g < ng; g++) {
                // Bottom ghost ← top interior
                full_G[k * nx_t * ny_total_g + g * nx_t + i] =
                    full_G[k * nx_t * ny_total_g + (gny + g) * nx_t + i];
                // Top ghost ← bottom interior
                full_G[k * nx_t * ny_total_g + (gny + ng + g) * nx_t + i] =
                    full_G[k * nx_t * ny_total_g + (ng + g) * nx_t + i];
            }
}

/**
 * Gather local fields to rank 0 and save binary file.
 * Binary format: int32[4] (nx, ny, nz, nghost) + float64[full array with ghosts]
 */
static bool gatherAndSave(const char* filename,
                            const double* d_G_local,
                            SimParams& p) {
    int nx_t = p.nx_total, ny_t = p.ny_total, nz_t = p.nz_total;
    int local_total = nx_t * ny_t * nz_t;
    int interior_size = nx_t * p.local_ny * nz_t;

    // D2H
    std::vector<double> h_local(local_total);
    CUDA_CHECK(cudaMemcpy(h_local.data(), d_G_local,
                           local_total * sizeof(double), cudaMemcpyDeviceToHost));

    // Pack interior rows
    std::vector<double> h_interior(interior_size);
    packLocalInterior(h_local.data(), h_interior.data(), p);

    // Gather sizes
    std::vector<int> recv_counts(p.num_procs);
    std::vector<int> displs(p.num_procs, 0);
    MPI_CHECK(MPI_Gather(&interior_size, 1, MPI_INT,
                           recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (p.rank == 0) {
        for (int r = 1; r < p.num_procs; r++)
            displs[r] = displs[r-1] + recv_counts[r-1];
    }

    int global_interior_size = (p.rank == 0)
        ? displs[p.num_procs-1] + recv_counts[p.num_procs-1] : 0;

    std::vector<double> global_interior;
    if (p.rank == 0) global_interior.resize(global_interior_size);

    MPI_CHECK(MPI_Gatherv(h_interior.data(), interior_size, MPI_DOUBLE,
                            global_interior.data(), recv_counts.data(),
                            displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (p.rank == 0) {
        int gny = p.global_ny;
        int ny_total_g = gny + 2 * p.nghost;
        int full_size = nx_t * ny_total_g * nz_t;

        std::vector<double> full_G(full_size);
        reconstructFullField(global_interior.data(), full_G.data(), p,
                              recv_counts.data(), displs.data());

        FILE* fp = fopen(filename, "wb");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return false; }

        int header[4] = {p.nx, gny, p.nz, p.nghost};
        fwrite(header, sizeof(int), 4, fp);
        fwrite(full_G.data(), sizeof(double), full_size, fp);
        fclose(fp);
    }
    return true;
}

//=============================================================================
// Reduction Helpers
//=============================================================================

static double computeLocalVolume(const double* d_G, const SimParams& p) {
    int local_total = p.nx_total * p.ny_total * p.nz_total;
    std::vector<double> h_G(local_total);
    CUDA_CHECK(cudaMemcpy(h_G.data(), d_G, local_total * sizeof(double),
                           cudaMemcpyDeviceToHost));

    double vol = 0.0;
    for (int k = p.nghost; k < p.nz + p.nghost; k++)
        for (int j = p.nghost; j < p.local_ny + p.nghost; j++)
            for (int i = p.nghost; i < p.nx + p.nghost; i++)
                if (h_G[idx(i, j, k, p.nx_total, p.ny_total)] < 0.0)
                    vol += p.dx * p.dy * p.dz;
    return vol;
}

static double computeLocalL2(const double* d_G, const double* d_G_ref,
                               const SimParams& p) {
    int total = p.nx_total * p.ny_total * p.nz_total;
    std::vector<double> hG(total), hR(total);
    CUDA_CHECK(cudaMemcpy(hG.data(), d_G, total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR.data(), d_G_ref, total*sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    int count = 0;
    for (int k = p.nghost; k < p.nz + p.nghost; k++)
        for (int j = p.nghost; j < p.local_ny + p.nghost; j++)
            for (int i = p.nghost; i < p.nx + p.nghost; i++) {
                int id = idx(i, j, k, p.nx_total, p.ny_total);
                double d = hG[id] - hR[id];
                sum += d * d;
                count++;
            }

    // Reduce across ranks
    double global_sum;
    int global_count;
    MPI_CHECK(MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    return sqrt(global_sum / global_count);
}

static double computeGlobalVolume(const double* d_G, const SimParams& p) {
    double local_vol = computeLocalVolume(d_G, p);
    double global_vol;
    MPI_CHECK(MPI_Allreduce(&local_vol, &global_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    return global_vol;
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[]) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Assign GPU: one GPU per rank (round-robin if fewer GPUs than ranks)
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));

    // --- Parameters ---
    SimParams params = initDefaultParams();
    setupMPIDomainDecomposition(params, rank, num_procs);

    if (params.dt == 0.0) {
        double max_vel = 2.0;
        params.dt = params.cfl * fmin(fmin(params.dx, params.dy), params.dz) / max_vel;
    }

    // --- Print info (rank 0 only) ---
    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        printf("========================================\n");
        printf("3D Level-Set Deformation Test (MPI+CUDA)\n");
        printf("========================================\n");
        printf("MPI processes: %d,  GPUs per node: %d\n", num_procs, num_devices);
        printf("GPU 0: %s\n", prop.name);
        printf("Global grid: %d x %d x %d\n", params.nx, params.global_ny, params.nz);
        printf("Domain decomposition: 1D in Y\n");
        printf("dt = %.6e,  T_final = %.4f\n\n", params.dt, params.t_final);
    }

    // Log local decomposition
    for (int r = 0; r < num_procs; r++) {
        if (rank == r)
            printf("  Rank %d: GPU %d, local_ny=%d, y_start=%d, neighbours=[%d,%d]\n",
                   rank, device_id, params.local_ny, params.y_start,
                   params.neighbor_below, params.neighbor_above);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (rank == 0) printf("\n");

    // --- Allocate device memory ---
    int local_total = params.nx_total * params.ny_total * params.nz_total;
    size_t bytes = local_total * sizeof(double);

    double *d_G, *d_G_new, *d_G_1, *d_G_2, *d_G_rhs, *d_G_initial;
    double *d_u, *d_v, *d_w;

    CUDA_CHECK(cudaMalloc(&d_G, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_1, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_2, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_rhs, bytes));
    CUDA_CHECK(cudaMalloc(&d_G_initial, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, bytes));

    // --- Halo buffers ---
    HaloBuffers halo;
    allocHaloBuffers(halo, params);

    // --- Initialise ---
    initSphereDeformationTest(d_G, d_u, d_v, d_w, params);
    applyBoundaryConditions(d_G, params, halo);
    applyVelocityBoundaryConditions(d_u, d_v, d_w, params, halo);
    copyInitialField(d_G, d_G_initial, local_total);

    double initial_volume = computeGlobalVolume(d_G, params);
    if (rank == 0) printf("Initial interface volume: %.6f\n\n", initial_volume);

    // Save initial field
    if (rank == 0) { system("mkdir -p output"); system("mkdir -p log"); }
    MPI_Barrier(MPI_COMM_WORLD);
    gatherAndSave("output/G_initial.bin", d_G, params);

    // --- Time loop ---
    if (rank == 0) printf("Starting simulation...\n\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    double current_time = 0.0;
    int step = 0;
    int max_steps = (int)(params.t_final / params.dt) + 100;

    while (current_time < params.t_final && step < max_steps) {
        if (current_time + params.dt > params.t_final)
            params.dt = params.t_final - current_time;

        // Update velocity
        params.current_time = current_time;
        updateDeformationVelocity(d_u, d_v, d_w, params);
        // applyVelocityBoundaryConditions(d_u, d_v, d_w, params, halo);

        // RK3 step
        rk3TimeStep(d_G, d_G_new, d_G_1, d_G_2, d_G_rhs,
                     d_u, d_v, d_w, params, halo);

        // Swap
        double* tmp = d_G; d_G = d_G_new; d_G_new = tmp;

        current_time += params.dt;
        step++;

        if (step % OUTPUT_INTERVAL == 0) {
            double l2  = computeLocalL2(d_G, d_G_initial, params);
            double vol = computeGlobalVolume(d_G, params);
            if (rank == 0)
                printf("Step %6d: t=%.6f  dt=%.3e  L2=%.6e  Vol=%.6f\n",
                       step, current_time, params.dt, l2, vol);

            // char fname[256];
            // snprintf(fname, sizeof(fname), "output/G_step_%06d.bin", step);
            // gatherAndSave(fname, d_G, params);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // --- Final results ---
    double final_l2  = computeLocalL2(d_G, d_G_initial, params);
    double final_vol = computeGlobalVolume(d_G, params);

    if (rank == 0) {
        printf("\n========================================\n");
        printf("Simulation Complete\n");
        printf("========================================\n");
        printf("Total steps: %d\n", step);
        printf("Final L2 Error: %.6e\n", final_l2);
        printf("Initial Volume: %.6f\n", initial_volume);
        printf("Final Volume:   %.6f\n", final_vol);
        printf("Volume Change:  %.4f%%\n",
               fabs(final_vol - initial_volume) / initial_volume * 100.0);
        printf("Wall time: %.3f s\n", elapsed);
        printf("Time per step: %.3f ms\n", elapsed * 1000.0 / step);
        printf("========================================\n");

        FILE* log_fp = fopen("log/simulation.log", "w");
        if (log_fp) {
            fprintf(log_fp, "========================================\n");
            fprintf(log_fp, "3D Level-Set (MPI+CUDA), %d ranks\n", num_procs);
            fprintf(log_fp, "========================================\n");
            fprintf(log_fp, "Grid: %d x %d x %d\n", params.nx, params.global_ny, params.nz);
            fprintf(log_fp, "dt: %.6e, T: %.4f\n", params.dt, params.t_final);
            fprintf(log_fp, "Steps: %d\n", step);
            fprintf(log_fp, "Final L2: %.6e\n", final_l2);
            fprintf(log_fp, "Volume: %.6f -> %.6f (%.4f%%)\n",
                    initial_volume, final_vol,
                    fabs(final_vol - initial_volume) / initial_volume * 100.0);
            fprintf(log_fp, "Wall time: %.3f s (%.3f ms/step)\n",
                    elapsed, elapsed * 1000.0 / step);
            fclose(log_fp);
        }
    }

    gatherAndSave("output/G_final.bin", d_G, params);

    // --- Cleanup ---
    freeHaloBuffers(halo);
    cudaFree(d_G); cudaFree(d_G_new);
    cudaFree(d_G_1); cudaFree(d_G_2); cudaFree(d_G_rhs);
    cudaFree(d_G_initial);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_w);

    MPI_Finalize();
    return 0;
}
