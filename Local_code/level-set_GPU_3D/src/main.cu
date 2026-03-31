/**
 * @file main.cu
 * @brief Main entry point for 3D Level-Set Deformation Test
 *
 * This program solves the level-set equation in 3D using:
 * - WENO-5 spatial discretization
 * - TVD RK3 time integration
 * - Deformation test velocity field
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

// Include all headers
#include "../include/config.cuh"
#include "../include/weno5.cuh"
#include "../include/initial_conditions.cuh"
#include "../include/rk3.cuh"
#include "../include/boundary.cuh"

//=============================================================================
// Simple IO Functions for 3D
//=============================================================================

/**
 * @brief Save 3D field to binary file
 */
bool saveFieldBinary(const char* filename, const double* h_G, const SimParams& params) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return false;
    }

    // Write header
    int header[4] = {params.nx, params.ny, params.nz, params.nghost};
    fwrite(header, sizeof(int), 4, fp);

    // Write data
    int total_size = params.nx_total * params.ny_total * params.nz_total;
    fwrite(h_G, sizeof(double), total_size, fp);

    fclose(fp);
    return true;
}

/**
 * @brief Compute interface volume (region where G < 0)
 */
double computeInterfaceVolume(const double* d_G, SimParams& params) {
    int total_size = params.nx_total * params.ny_total * params.nz_total;
    double* h_G = new double[total_size];
    CUDA_CHECK(cudaMemcpy(h_G, d_G, total_size * sizeof(double), cudaMemcpyDeviceToHost));

    double volume = 0.0;
    for (int k = params.nghost; k < params.nz + params.nghost; k++) {
        for (int j = params.nghost; j < params.ny + params.nghost; j++) {
            for (int i = params.nghost; i < params.nx + params.nghost; i++) {
                int index = idx(i, j, k, params.nx_total, params.ny_total);
                if (h_G[index] < 0.0) {
                    volume += params.dx * params.dy * params.dz;
                }
            }
        }
    }

    delete[] h_G;
    return volume;
}

/**
 * @brief Compute L2 error
 */
double computeL2Error(const double* d_G, const double* d_G_ref, SimParams& params) {
    int total_size = params.nx_total * params.ny_total * params.nz_total;
    double* h_G = new double[total_size];
    double* h_G_ref = new double[total_size];

    CUDA_CHECK(cudaMemcpy(h_G, d_G, total_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_G_ref, d_G_ref, total_size * sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    int count = 0;

    for (int k = params.nghost; k < params.nz + params.nghost; k++) {
        for (int j = params.nghost; j < params.ny + params.nghost; j++) {
            for (int i = params.nghost; i < params.nx + params.nghost; i++) {
                int index = idx(i, j, k, params.nx_total, params.ny_total);
                double diff = h_G[index] - h_G_ref[index];
                sum += diff * diff;
                count++;
            }
        }
    }

    delete[] h_G;
    delete[] h_G_ref;

    return sqrt(sum / count);
}

/**
 * @brief Print progress
 */
void printProgress(int step, double t, double dt, double l2_error, double volume) {
    printf("Step %6d: t = %.6f, dt = %.6e, L2 error = %.6e, Volume = %.6f\n",
           step, t, dt, l2_error, volume);
}

/**
 * @brief Print CUDA device info
 */
void printCUDAInfo() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("CUDA Device: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("\n");
}

/**
 * @brief Print simulation parameters
 */
void printParameters(const SimParams& params) {
    printf("========================================\n");
    printf("3D Level-Set Deformation Test\n");
    printf("========================================\n");
    printf("Grid:\n");
    printf("  Interior points: %d x %d x %d\n", params.nx, params.ny, params.nz);
    printf("  Total size: %d x %d x %d\n", params.nx_total, params.ny_total, params.nz_total);
    printf("  Domain: [%.2f, %.2f] x [%.2f, %.2f] x [%.2f, %.2f]\n",
           params.x_min, params.x_max, params.y_min, params.y_max, params.z_min, params.z_max);
    printf("  Grid spacing: dx=%.6f, dy=%.6f, dz=%.6f\n", params.dx, params.dy, params.dz);
    printf("\nTime integration:\n");
    printf("  CFL number: %.4f\n", params.cfl);
    printf("  Time step: dt = %.6e\n", params.dt);
    printf("  Final time: T = %.4f\n", params.t_final);
    printf("========================================\n\n");
}

//=============================================================================
// Main Simulation Function
//=============================================================================

void runDeformationTest() {
    printf("========================================\n");
    printf("3D Level-Set Deformation Test\n");
    printf("========================================\n\n");

    printCUDAInfo();

    // Create log directory and open log file
    system("mkdir -p log");
    FILE* log_fp = fopen("log/simulation.log", "w");
    if (log_fp) {
        fprintf(log_fp, "========================================\n");
        fprintf(log_fp, "3D Level-Set Deformation Test\n");
        fprintf(log_fp, "========================================\n\n");
    }

    // Initialize parameters
    SimParams params = initDefaultParams();

    // Calculate time step
    // If DT is set in config.cuh (non-zero), it will use that value
    // If DT is 0.0, it will calculate based on CFL and max velocity
    if (params.dt == 0.0) {
        // For deformation test, we need to consider max velocity magnitude
        // Max velocity is about 2.0 (from the velocity field definition)
        double max_vel = 2.0;
        params.dt = params.cfl * fmin(fmin(params.dx, params.dy), params.dz) / max_vel;
    }
    // Otherwise, params.dt is already set from DT in config.cuh

    printParameters(params);

    // Log parameters
    if (log_fp) {
        fprintf(log_fp, "Grid: %d x %d x %d\n", params.nx, params.ny, params.nz);
        fprintf(log_fp, "dt: %.6e, T: %.4f\n\n", params.dt, params.t_final);
    }

    // Allocate memory
    int total_size = params.nx_total * params.ny_total * params.nz_total;
    size_t bytes = total_size * sizeof(double);

    printf("Allocating memory: %d x %d x %d = %d cells (%.2f MB)\n",
           params.nx_total, params.ny_total, params.nz_total, total_size,
           bytes * 7 / (1024.0 * 1024.0));

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

    double* h_G = new double[total_size];

    // Initialize test case
    printf("Initializing sphere deformation test...\n");
    initSphereDeformationTest(d_G, d_u, d_v, d_w, params);
    applyBoundaryConditions(d_G, params);
    applyVelocityBoundaryConditions(d_u, d_v, d_w, params);

    // Store initial field
    CUDA_CHECK(cudaMemcpy(d_G_initial, d_G, bytes, cudaMemcpyDeviceToDevice));

    // Calculate initial volume
    double initial_volume = computeInterfaceVolume(d_G, params);
    printf("Initial interface volume: %.6f\n\n", initial_volume);

    // Save initial field
    system("mkdir -p output");
    system("mkdir -p log");
    CUDA_CHECK(cudaMemcpy(h_G, d_G, bytes, cudaMemcpyDeviceToHost));
    saveFieldBinary("output/G_initial.bin", h_G, params);
    printf("Saved initial field to output/G_initial.bin\n\n");

    // Main time loop
    printf("Starting simulation...\n");
    printf("Target time: %.4f, dt = %.6e\n\n", params.t_final, params.dt);

    auto start_time = std::chrono::high_resolution_clock::now();

    double current_time = 0.0;
    int step = 0;
    int max_steps = (int)(params.t_final / params.dt) + 100;
    int output_interval = 50;

    while (current_time < params.t_final && step < max_steps) {
        // Adjust final step
        if (current_time + params.dt > params.t_final) {
            params.dt = params.t_final - current_time;
        }

        // Update velocity field for current time
        params.current_time = current_time;
        updateDeformationVelocity(d_u, d_v, d_w, params);
        applyVelocityBoundaryConditions(d_u, d_v, d_w, params);

        // RK3 time step
        rk3TimeStep(d_G, d_G_new, d_G_1, d_G_2, d_G_rhs,
                    d_u, d_v, d_w, params, applyBoundaryConditions);

        // Swap pointers
        double* temp = d_G;
        d_G = d_G_new;
        d_G_new = temp;

        current_time += params.dt;
        step++;

        // Output progress
        if (step % output_interval == 0) {
            double l2_error = computeL2Error(d_G, d_G_initial, params);
            double volume = computeInterfaceVolume(d_G, params);
            printProgress(step, current_time, params.dt, l2_error, volume);

            // Save snapshot
            CUDA_CHECK(cudaMemcpy(h_G, d_G, bytes, cudaMemcpyDeviceToHost));
            char filename[256];
            snprintf(filename, sizeof(filename), "output/G_step_%06d.bin", step);
            saveFieldBinary(filename, h_G, params);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Final results
    printf("\n");
    double final_l2_error = computeL2Error(d_G, d_G_initial, params);
    double final_volume = computeInterfaceVolume(d_G, params);

    printf("\n========================================\n");
    printf("Simulation Complete\n");
    printf("========================================\n");
    printf("Total steps: %d\n", step);
    printf("Final L2 Error: %.6e\n", final_l2_error);
    printf("Initial Volume: %.6f\n", initial_volume);
    printf("Final Volume: %.6f\n", final_volume);
    printf("Volume Conservation Error: %.4f%%\n",
           fabs(final_volume - initial_volume) / initial_volume * 100.0);
    printf("Wall time: %.3f s\n", elapsed);
    printf("Time per step: %.3f ms\n", elapsed * 1000.0 / step);
    printf("========================================\n");

    // Write to log file
    if (log_fp) {
        fprintf(log_fp, "\nSimulation Complete\n");
        fprintf(log_fp, "Total steps: %d\n", step);
        fprintf(log_fp, "Final L2 Error: %.6e\n", final_l2_error);
        fprintf(log_fp, "Initial Volume: %.6f\n", initial_volume);
        fprintf(log_fp, "Final Volume: %.6f\n", final_volume);
        fprintf(log_fp, "Volume Conservation Error: %.4f%%\n",
               fabs(final_volume - initial_volume) / initial_volume * 100.0);
        fprintf(log_fp, "Wall time: %.3f s\n", elapsed);
        fprintf(log_fp, "Time per step: %.3f ms\n", elapsed * 1000.0 / step);
        fclose(log_fp);
    }

    // Save final field
    CUDA_CHECK(cudaMemcpy(h_G, d_G, bytes, cudaMemcpyDeviceToHost));
    saveFieldBinary("output/G_final.bin", h_G, params);
    printf("\nSaved final field to output/G_final.bin\n");

    // Cleanup
    cudaFree(d_G);
    cudaFree(d_G_new);
    cudaFree(d_G_1);
    cudaFree(d_G_2);
    cudaFree(d_G_rhs);
    cudaFree(d_G_initial);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    delete[] h_G;

    printf("\nSimulation complete.\n");
}

//=============================================================================
// Main Entry Point
//=============================================================================

int main(int argc, char* argv[]) {
    runDeformationTest();
    return 0;
}
