#include "config_cuda.cuh"
#include "field_cuda.cuh"
#include "weno5_kernels.cuh"
#include "rk3_solver.cuh"

#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <string>
#include <iomanip>
#include <sstream>

// ============================================================
// File I/O functions
// ============================================================

void save_G_to_bin(const double* h_G, const std::string& filename) {
    std::filesystem::path path(filename);
    std::filesystem::path dir = path.parent_path();

    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::cerr << "Error: Directory does not exist: " << dir << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Save internal domain only (excluding ghost cells)
    for (int j = GHOST; j < NY + GHOST; ++j) {
        file.write(reinterpret_cast<const char*>(&h_G[idx2d(j, GHOST)]), NX * sizeof(double));
    }
    file.close();
}

// ============================================================
// Initialization kernel
// ============================================================

__global__ void kernel_initialize_fields(
    double* G,
    double* G_initial,
    double* u,
    double* v,
    double R,
    double slope,
    double X0,
    double Y0
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX_TOTAL || j >= NY_TOTAL) return;

    int idx = idx2d(j, i);
    double x = (i - GHOST) * DX;
    double y = (j - GHOST) * DY;

    double distance;
    if (x < X0)
        distance = slope * (x - (X0 - R)) - (y - Y0);
    else
        distance = -slope * (x - (X0 + R)) - (y - Y0);

    G[idx] = -distance;
    G_initial[idx] = -distance;
    u[idx] = u0;
    v[idx] = v0;
}

// ============================================================
// Main function
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " freq" << std::endl;
        return 1;
    }

    int freq = std::atoi(argv[1]);

    // Print GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Grid size: " << NX << " x " << NY << std::endl;

    // Create log directory and file
    std::filesystem::create_directories("./log");
    std::ofstream log_file("./log/run_" + std::to_string(freq) + "Hz_gpu.log");
    if (!log_file) {
        std::cerr << "Cannot open the log file." << std::endl;
        return 1;
    }

    auto log = [&](const auto& msg) {
        std::cout << msg << std::flush;
        log_file << msg << std::flush;
    };

    auto now_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm now_tm = *std::localtime(&now_time_t);
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%F %T");
    log("Start : " + oss.str() + "\n");

    std::filesystem::create_directories("./results/" + std::to_string(freq) + "Hz/");

    // ========================================================
    // Allocate device memory
    // ========================================================

    size_t field_size = NX_TOTAL * NY_TOTAL * sizeof(double);

    double *d_G, *d_G1, *d_G2, *d_G_initial;
    double *d_u, *d_v, *d_rhs;

    CUDA_CHECK(cudaMalloc(&d_G, field_size));
    CUDA_CHECK(cudaMalloc(&d_G1, field_size));
    CUDA_CHECK(cudaMalloc(&d_G2, field_size));
    CUDA_CHECK(cudaMalloc(&d_G_initial, field_size));
    CUDA_CHECK(cudaMalloc(&d_u, field_size));
    CUDA_CHECK(cudaMalloc(&d_v, field_size));
    CUDA_CHECK(cudaMalloc(&d_rhs, field_size));

    // Host buffer for output
    double* h_G = new double[NX_TOTAL * NY_TOTAL];

    // ========================================================
    // Initialize fields on GPU
    // ========================================================

    double R = 0.005;
    double flame_angle = 30.0;
    double slope = std::tan((90.0 - flame_angle) * M_PI / 180.0);
    double X0 = 0.01;
    double Y0 = 0.0;

    dim3 block_init(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_init((NX_TOTAL + block_init.x - 1) / block_init.x,
                   (NY_TOTAL + block_init.y - 1) / block_init.y);

    kernel_initialize_fields<<<grid_init, block_init>>>(
        d_G, d_G_initial, d_u, d_v, R, slope, X0, Y0
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply initial boundary conditions
    apply_anchored_bc_gpu(d_G, d_G_initial);
    apply_periodic_gpu(d_G);

    // Save initial field
    CUDA_CHECK(cudaMemcpy(h_G, d_G, field_size, cudaMemcpyDeviceToHost));
    save_G_to_bin(h_G, "./results/" + std::to_string(freq) + "Hz/itr_G_field_0.bin");

    // ========================================================
    // Time integration loop
    // ========================================================

    auto start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;
    int save_idx = 1;

    dim3 block_vel(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_vel((NX_TOTAL + block_vel.x - 1) / block_vel.x,
                  (NY_TOTAL + block_vel.y - 1) / block_vel.y);

    while (t < FINAL_TIME) {
        // Apply boundary conditions
        apply_anchored_bc_gpu(d_G, d_G_initial);
        apply_periodic_gpu(d_G);

        // Update velocity field
        kernel_update_velocity<<<grid_vel, block_vel>>>(d_u, d_v, t, freq);
        CUDA_CHECK(cudaGetLastError());

        // RK3 time integration
        RK3_step_gpu(d_G, d_G1, d_G2, d_u, d_v, d_rhs, d_G_initial, DX, DY, DT, SL);

        // Apply boundary conditions after RK3
        apply_periodic_gpu(d_G);
        apply_anchored_bc_gpu(d_G, d_G_initial);

        t += DT;
        step++;

        if (step % save_interval == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            log("Step " + std::to_string(step) + ", t = " + std::to_string(t) + "\n");

            CUDA_CHECK(cudaMemcpy(h_G, d_G, field_size, cudaMemcpyDeviceToHost));
            save_G_to_bin(h_G, "./results/" + std::to_string(freq) +
                         "Hz/itr_G_field_" + std::to_string(save_idx) + ".bin");
            save_idx++;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    log("\nSimulation finished at t = " + std::to_string(t) + "\n");
    log("Elapsed time: " + std::to_string(elapsed_seconds.count()) + " seconds\n");

    // ========================================================
    // Cleanup
    // ========================================================

    delete[] h_G;
    cudaFree(d_G);
    cudaFree(d_G1);
    cudaFree(d_G2);
    cudaFree(d_G_initial);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_rhs);

    return 0;
}
