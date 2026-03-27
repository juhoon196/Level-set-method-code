/**
 * @file solver.cpp
 * @brief G-equation Level-Set Solver class implementation (MPI 3D version)
 */

#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "../include/config.h"
#include "../include/weno5.h"
#include "../include/rk3.h"
#include "../include/reinitialization.h"
#include "../include/boundary.h"
#include "../include/initial_conditions.h"
#include "../include/io.h"

//=============================================================================
// G-Equation Solver Class (MPI 3D Version)
//=============================================================================

class GEquationSolver3D {
public:
    SimParams params;

    // Local arrays
    double* G;
    double* G_new;
    double* G_1;
    double* G_2;
    double* G_rhs;
    double* G_initial;
    double* u;
    double* v;
    double* w;

    // Reinitialization arrays
    double* G_reinit_temp;
    double* G_reinit_0;
    double* r_tilde;
    int* interface_flag;

    // State
    double current_time;
    int current_step;
    bool is_initialized;

    char output_dir[256];

    GEquationSolver3D() : is_initialized(false), current_time(0.0), current_step(0) {
        params = initDefaultParams();
        strcpy(output_dir, "./output");

        G = nullptr; G_new = nullptr; G_1 = nullptr; G_2 = nullptr;
        G_rhs = nullptr; G_initial = nullptr;
        u = nullptr; v = nullptr; w = nullptr;
        G_reinit_temp = nullptr; G_reinit_0 = nullptr;
        r_tilde = nullptr; interface_flag = nullptr;
    }

    ~GEquationSolver3D() {
        cleanup();
    }

    void setupMPI(int rank, int num_procs) {
        setupMPIDomainDecomposition(params, rank, num_procs);
    }

    void setParameters(const SimParams& new_params) {
        params = new_params;
    }

    void setOutputDirectory(const char* dir) {
        strcpy(output_dir, dir);
    }

    bool allocateMemory() {
        long long local_total_size = (long long)params.nx_total * params.local_ny_total * params.nz_total;
        size_t bytes = local_total_size * sizeof(double);

        if (params.rank == 0) {
            printf("Allocating memory: %d x %d x %d = %lld cells per process (%.2f MB)\n",
                   params.nx_total, params.local_ny_total, params.nz_total,
                   local_total_size, bytes * 13 / (1024.0 * 1024.0));
        }

        G             = new double[local_total_size]();
        G_new         = new double[local_total_size]();
        G_1           = new double[local_total_size]();
        G_2           = new double[local_total_size]();
        G_rhs         = new double[local_total_size]();
        G_initial     = new double[local_total_size]();
        u             = new double[local_total_size]();
        v             = new double[local_total_size]();
        w             = new double[local_total_size]();
        G_reinit_temp = new double[local_total_size]();
        G_reinit_0    = new double[local_total_size]();
        r_tilde       = new double[local_total_size]();
        interface_flag = new int[local_total_size]();

        if (params.rank == 0) {
            printf("Memory allocation complete.\n");
        }

        return true;
    }

    void cleanup() {
        delete[] G; G = nullptr;
        delete[] G_new; G_new = nullptr;
        delete[] G_1; G_1 = nullptr;
        delete[] G_2; G_2 = nullptr;
        delete[] G_rhs; G_rhs = nullptr;
        delete[] G_initial; G_initial = nullptr;
        delete[] u; u = nullptr;
        delete[] v; v = nullptr;
        delete[] w; w = nullptr;
        delete[] G_reinit_temp; G_reinit_temp = nullptr;
        delete[] G_reinit_0; G_reinit_0 = nullptr;
        delete[] r_tilde; r_tilde = nullptr;
        delete[] interface_flag; interface_flag = nullptr;

        is_initialized = false;
    }

    bool initializeSphereTest() {
        if (!allocateMemory()) return false;

        params.dt = calculateTimeStep(params);

        initSphereTestCase(G, u, v, w, params);
        applyBoundaryConditions3D(G, params);
        applyVelocityBoundaryConditions3D(u, v, w, params);

        long long total = (long long)params.nx_total * params.local_ny_total * params.nz_total;
        copyInitialField(G, G_initial, total);

        current_time = 0.0;
        current_step = 0;
        is_initialized = true;

        printParameters3D(params);
        if (params.rank == 0) {
            printf("Initialized Sphere Advection Test Case\n\n");
        }

        return true;
    }

    bool initializeZalesakTest() {
        if (!allocateMemory()) return false;

        params.dt = calculateTimeStep(params);

        initZalesak3DTestCase(G, u, v, w, params);
        applyBoundaryConditions3D(G, params);
        applyVelocityBoundaryConditions3D(u, v, w, params);

        long long total = (long long)params.nx_total * params.local_ny_total * params.nz_total;
        copyInitialField(G, G_initial, total);

        current_time = 0.0;
        current_step = 0;
        is_initialized = true;

        printParameters3D(params);

        return true;
    }

    void step() {
        if (!is_initialized) {
            fprintf(stderr, "Error: Solver not initialized\n");
            return;
        }

        rk3TimeStep3D(G, G_new, G_1, G_2, G_rhs, u, v, w, params);

        double* temp = G;
        G = G_new;
        G_new = temp;

        if (params.enable_reinit &&
            current_step > 0 &&
            current_step % params.reinit_interval == 0) {
            reinitializeWithSwap3D(&G, &G_reinit_temp, G_reinit_0,
                                    r_tilde, interface_flag, params);
        }

        current_time += params.dt;
        current_step++;
    }

    void run() {
        if (!is_initialized) {
            fprintf(stderr, "Error: Solver not initialized\n");
            return;
        }

        if (params.rank == 0) {
            printf("Starting simulation...\n");
            printf("Target time: %.4f, dt = %.6e, estimated steps: %d\n\n",
                   params.t_final, params.dt, (int)(params.t_final / params.dt));
        }

        double initial_vol = computeInterfaceVolume(G, params);
        if (params.rank == 0) {
            printf("Initial interface volume: %.6f\n\n", initial_vol);
        }

        while (current_time < params.t_final && current_step < MAX_STEPS) {
            if (current_time + params.dt > params.t_final) {
                params.dt = params.t_final - current_time;
            }

            step();

            if (current_step % OUTPUT_INTERVAL == 0) {
                double l2_error = computeL2Error3D(G, G_initial, params);
                double vol = computeInterfaceVolume(G, params);
                printProgress3D(current_step, current_time, params.dt, l2_error, vol, params.rank);
            }
        }

        double final_l2_error = computeL2Error3D(G, G_initial, params);
        double final_vol = computeInterfaceVolume(G, params);
        double shape_error = computeMeanShapeError3D(G, G_initial, params);

        printSummary3D(final_l2_error, shape_error, initial_vol, final_vol, params.rank);
    }

    bool saveField(const char* filename) {
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", output_dir, filename);
        return saveFieldBinary3D(fullpath, G, params);
    }

    bool saveFieldAsVTK(const char* filename) {
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", output_dir, filename);
        return saveFieldVTK3D(fullpath, G, params);
    }

    double getL2Error() {
        return computeL2Error3D(G, G_initial, params);
    }

    double getInterfaceVolume() {
        return computeInterfaceVolume(G, params);
    }

    double getCurrentTime() const { return current_time; }
    int getCurrentStep() const { return current_step; }
};

//=============================================================================
// Convenience Functions
//=============================================================================

void runSimulation3D(int test_case, bool save_output, int rank, int num_procs) {
    GEquationSolver3D solver;
    solver.setupMPI(rank, num_procs);

    bool init_success = false;
    switch (test_case) {
        case 0:
            init_success = solver.initializeSphereTest();
            break;
        case 1:
            init_success = solver.initializeZalesakTest();
            break;
        default:
            if (rank == 0) fprintf(stderr, "Unknown test case: %d\n", test_case);
            return;
    }

    if (!init_success) {
        if (rank == 0) fprintf(stderr, "Initialization failed\n");
        return;
    }

    if (save_output) solver.saveField("G_initial.bin");

    solver.run();

    if (save_output) {
        solver.saveField("G_final.bin");
        solver.saveFieldAsVTK("G_final.vtk");
    }
}
