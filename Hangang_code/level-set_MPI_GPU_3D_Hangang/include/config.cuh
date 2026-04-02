/**
 * @file config.cuh
 * @brief Configuration for G-equation Level-Set solver (MPI + CUDA multi-GPU)
 *
 * Y-direction 1D domain decomposition across MPI ranks.
 * Each rank owns one GPU and a horizontal slab of the domain.
 *
 * Key design: SimParams.ny / ny_total always refer to the LOCAL partition.
 * Kernels see only local dimensions and work unmodified from single-GPU.
 * global_ny stores the full domain size for I/O and coordinate mapping.
 */

#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

//=============================================================================
// Grid Configuration (GLOBAL domain)
//=============================================================================

constexpr int NX = 201;
constexpr int NY = 201;
constexpr int NZ = 201;

constexpr int NGHOST = 3;

constexpr int NX_TOTAL = NX + 2 * NGHOST;
constexpr int NZ_TOTAL = NZ + 2 * NGHOST;

// Physical domain
constexpr double DOMAIN_X_MIN = 0.0;
constexpr double DOMAIN_X_MAX = 1.0;
constexpr double DOMAIN_Y_MIN = 0.0;
constexpr double DOMAIN_Y_MAX = 1.0;
constexpr double DOMAIN_Z_MIN = 0.0;
constexpr double DOMAIN_Z_MAX = 1.0;

// Grid spacing (global, identical on every rank)
constexpr double DX = (DOMAIN_X_MAX - DOMAIN_X_MIN) / (NX - 1);
constexpr double DY = (DOMAIN_Y_MAX - DOMAIN_Y_MIN) / (NY - 1);
constexpr double DZ = (DOMAIN_Z_MAX - DOMAIN_Z_MIN) / (NZ - 1);

//=============================================================================
// Physical Parameters
//=============================================================================

constexpr double S_L = 0.0;
constexpr double U_CONST = 0.0;
constexpr double V_CONST = 0.0;
constexpr double W_CONST = 0.0;

//=============================================================================
// Time Integration Parameters
//=============================================================================

constexpr double DT = 0.0;
constexpr double CFL = 0.2;
constexpr double T_FINAL = 1.5;
constexpr int MAX_STEPS = 15;// 1000000;
constexpr int OUTPUT_INTERVAL = 50;

//=============================================================================
// Reinitialization Parameters
//=============================================================================

constexpr bool ENABLE_REINIT = false;
constexpr int REINIT_INTERVAL = 10;
constexpr int REINIT_ITERATIONS = 2;
constexpr double REINIT_DTAU_FACTOR = 0.25;
constexpr double REINIT_BETA = 0.5;
constexpr double EPSILON = 1.0e-10;
constexpr double INTERFACE_BAND = 2.0;

//=============================================================================
// CUDA Configuration
//=============================================================================

constexpr int BLOCK_SIZE_X = 8;
constexpr int BLOCK_SIZE_Y = 8;
constexpr int BLOCK_SIZE_Z = 8;

//=============================================================================
// Error-Checking Macros
//=============================================================================

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                          \
        }                                                                     \
    } while (0)

#define MPI_CHECK(call)                                                       \
    do {                                                                      \
        int err = call;                                                       \
        if (err != MPI_SUCCESS) {                                             \
            char estr[MPI_MAX_ERROR_STRING]; int elen;                        \
            MPI_Error_string(err, estr, &elen);                               \
            fprintf(stderr, "MPI error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, estr);                                \
            MPI_Abort(MPI_COMM_WORLD, err);                                   \
        }                                                                     \
    } while (0)

//=============================================================================
// Simulation Parameters
//=============================================================================

struct SimParams {
    // Grid — LOCAL (what kernels see)
    int nx, ny, nz;                 // ny = local_ny for this rank
    int nx_total, ny_total, nz_total; // ny_total = local_ny + 2*NGHOST
    int nghost;
    double dx, dy, dz;
    double x_min, x_max, y_min, y_max, z_min, z_max;

    // MPI domain decomposition (1D in Y)
    int rank, num_procs;
    int global_ny;                  // total interior Y points across all ranks
    int local_ny;                   // interior Y points for this rank (== ny)
    int y_start;                    // global y-index of first interior row
    int neighbor_below, neighbor_above; // periodic wrap-around

    // Physical
    double s_l, u_const, v_const, w_const;

    // Time
    double dt, t_final, cfl, current_time;

    // Reinitialization
    bool enable_reinit;
    int reinit_interval, reinit_iterations;
    double reinit_dtau, reinit_beta;

    // Numerical
    double epsilon, interface_band;
};

//=============================================================================
// Initialization Helpers
//=============================================================================

inline SimParams initDefaultParams() {
    SimParams p{};

    // Grid — will be overwritten by setupMPIDomainDecomposition
    p.nx = NX;  p.ny = NY;  p.nz = NZ;
    p.nx_total = NX_TOTAL;
    p.ny_total = NY + 2 * NGHOST;
    p.nz_total = NZ_TOTAL;
    p.nghost   = NGHOST;
    p.dx = DX;  p.dy = DY;  p.dz = DZ;
    p.x_min = DOMAIN_X_MIN; p.x_max = DOMAIN_X_MAX;
    p.y_min = DOMAIN_Y_MIN; p.y_max = DOMAIN_Y_MAX;
    p.z_min = DOMAIN_Z_MIN; p.z_max = DOMAIN_Z_MAX;

    p.rank = 0; p.num_procs = 1;
    p.global_ny = NY;
    p.local_ny  = NY;
    p.y_start   = 0;
    p.neighbor_below = MPI_PROC_NULL;
    p.neighbor_above = MPI_PROC_NULL;

    p.s_l = S_L; p.u_const = U_CONST; p.v_const = V_CONST; p.w_const = W_CONST;

    p.dt = DT; p.t_final = T_FINAL; p.cfl = CFL; p.current_time = 0.0;

    p.enable_reinit     = ENABLE_REINIT;
    p.reinit_interval   = REINIT_INTERVAL;
    p.reinit_iterations = REINIT_ITERATIONS;
    p.reinit_dtau       = REINIT_DTAU_FACTOR * fmin(fmin(DX, DY), DZ);
    p.reinit_beta       = REINIT_BETA;

    p.epsilon        = EPSILON;
    p.interface_band = INTERFACE_BAND;

    return p;
}

/**
 * @brief 1D Y-direction domain decomposition (periodic wrap-around)
 */
inline void setupMPIDomainDecomposition(SimParams& p, int rank, int num_procs) {
    p.rank = rank;
    p.num_procs = num_procs;
    p.global_ny = NY;

    int base  = NY / num_procs;
    int extra = NY % num_procs;

    p.local_ny = (rank < extra) ? base + 1 : base;
    p.y_start  = (rank < extra) ? rank * (base + 1)
                                : extra * (base + 1) + (rank - extra) * base;

    // LOCAL grid that kernels will see
    p.ny       = p.local_ny;
    p.ny_total = p.local_ny + 2 * NGHOST;

    // Periodic neighbours
    p.neighbor_below = (rank > 0)              ? rank - 1 : num_procs - 1;
    p.neighbor_above = (rank < num_procs - 1)  ? rank + 1 : 0;
}

inline double calculateTimeStep(const SimParams& p) {
    if (p.dt > 0.0) return p.dt;
    double max_speed = fabs(p.u_const) + fabs(p.v_const) + fabs(p.w_const) + p.s_l;
    if (max_speed < p.epsilon) max_speed = 2.0;
    return p.cfl * fmin(fmin(p.dx, p.dy), p.dz) / max_speed;
}

//=============================================================================
// Grid-Dimension Helpers (parameterised for local partitions)
//=============================================================================

inline dim3 getGridDim(const SimParams& p) {
    return dim3((p.nx_total + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (p.ny_total + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
                (p.nz_total + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z);
}

inline dim3 getBlockDim() {
    return dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
}

//=============================================================================
// Index Helpers
//=============================================================================

__host__ __device__ inline int idx(int i, int j, int k,
                                    int nx_total, int ny_total) {
    return k * (nx_total * ny_total) + j * nx_total + i;
}

/** Coordinate conversion with global y_start offset */
__host__ __device__ inline void indexToCoord(int i, int j, int k,
                                              double x_min, double y_min, double z_min,
                                              double dx, double dy, double dz,
                                              int nghost, int y_start,
                                              double& x, double& y, double& z) {
    x = x_min + (i - nghost) * dx;
    y = y_min + (y_start + j - nghost) * dy;
    z = z_min + (k - nghost) * dz;
}

#endif // CONFIG_CUH
