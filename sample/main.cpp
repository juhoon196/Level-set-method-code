#include "config.hpp"
#include "field.hpp"
#include "weno5_rk3.hpp"
#include "first_order_upwind.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <chrono> 
#include <omp.h>
#include <cstdlib>  // atoi
#include <string>   // std::to_string
#include <iomanip>  
#include <sstream>   

void load_field_from_bin(std::vector<std::vector<double>>& field, const std::string& filename, int nx, int ny) {
    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: File not found: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int j = 2; j <= ny+1; ++j) {
        file.read(reinterpret_cast<char*>(&field[j][2]), nx * sizeof(double));
    }
    file.close();
}

void save_G_to_bin(const std::vector<std::vector<double>>& G, const std::string& filename) {
    std::filesystem::path path(filename);
    std::filesystem::path dir = path.parent_path();

    if (!std::filesystem::exists(dir)) {
        std::cerr << "Error: Directory does not exist: " << dir << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::ofstream file(filename, std::ios::binary);
    int ny = G.size();        // = NY + 6
    int nx = G[0].size();     // = NX + 6
    // 내부 도메인만 저장 (ghost 제외)
    for (int j = 3; j <= ny-4; ++j) {
        file.write(reinterpret_cast<const char*>(&G[j][3]), (nx-6)*sizeof(double));
    }
    file.close();
}

int main(int argc, char* argv[]) {
    omp_set_num_threads(num_threads);
    std::cout << "OMP_NUM_THREADS = " << omp_get_max_threads() << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " freq" << std::endl;
        return 1;
    }
    
    int freq = std::atoi(argv[1]);

    /* ----‑‑‑‑‑ 로그 초기화 ---- */
    std::filesystem::create_directories("./log");
    std::ofstream log_file("./log/run_" + std::to_string(freq) + "Hz.log");
    if (!log_file) {
        std::cerr << "❌  cannot open the log file." << std::endl;
        return 1;
    }

    // “터미널 + 파일”로 동시에 찍는 작은 헬퍼 람다
    auto log = [&](const auto& msg) {
        std::cout << msg << std::flush;
        log_file  << msg << std::flush;
    };

    // 실행 정보 한 줄 기록 (날짜/시간 포함)
    auto now_time_t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm now_tm   = *std::localtime(&now_time_t);

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%F %T");   // "2025-05-07 15:58:00"
    log("Start : " + oss.str() + "\n");

    std::filesystem::create_directories("./results/" + std::to_string(freq) + "Hz/");

    Field2D G(NX, NY), u(NX, NY), v(NX, NY);
    Field2D G_initial(NX, NY);  // 초기 G를 따로 저장

    // =========================
    // 초기화 (Bunsen flame: distance function)
    // =========================
    double R = 0.005;  // base 반폭
    double flame_angle = 30.0;  // degrees
    double slope = std::tan((90.0 - flame_angle) * M_PI / 180.0); // cotangent(60도)
    double X0 = 0.01;  // **정확히 0.01로 중심 고정**
    double Y0 = 0.0; // anchoring y 위치
    
    for (int j = 0; j < NY+6; ++j) {
        for (int i = 0; i < NX+6; ++i) {
            double x = (i-3) * DX;  
            double y = (j-3) * DY;
            double distance;
            if (x < X0)
                distance = slope * (x - (X0 - R)) - (y - Y0);
            else
                distance = -slope * (x - (X0 + R)) - (y - Y0);
    
            G.data[j][i]        = -distance;
            G_initial.data[j][i]= -distance;
            u.data[j][i]        = u0;
            v.data[j][i]        = v0;
        }
    }

    // Save initial_G_field
    save_G_to_bin(G.data, "./results/" + std::to_string(freq) + "Hz/itr_G_field_0.bin");

    auto start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;
    int save_idx = 1;
    while (t < FINAL_TIME) {
        G.apply_anchored_bc(G_initial);
        G.apply_periodic();

    
        for (int j = 0; j < NY+6; ++j) {
            for (int i = 0; i < NX+6; ++i) {
                u.data[j][i] = u0 + epsilon_u*v0*cos(freq*2*M_PI*(t));
                v.data[j][i] = v0 + epsilon_v*v0*cos(freq*2*M_PI*(t));
            }
        }

        // compute_first_order_upwind(G.data, u.data, v.data, DX, DY, DT, SL);
        RK3_step(G, u, v, DX, DY, DT, SL, G_initial);
        G.apply_periodic();
        G.apply_anchored_bc(G_initial);

        // if (step % reinit_interval == 0) { 
        //     G.reinitialize(G_initial, DX, DY, reinit_itr);
        // }
        
        t += DT;
        step++;
        
        if (step % save_interval == 0) { 
            log("Step " + std::to_string(step) + ", t = " + std::to_string(t) + "\n");

            save_G_to_bin(G.data,
                "./results/" + std::to_string(freq) +
                "Hz/itr_G_field_" + std::to_string(save_idx) + ".bin");
            save_idx++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    log("\nSimulation finished at t = " + std::to_string(t) + "\n");
    log("Elapsed time: " + std::to_string(elapsed_seconds.count()) + " seconds\n");


    
    return 0;
}


