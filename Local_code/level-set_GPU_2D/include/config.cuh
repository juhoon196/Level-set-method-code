/**
 * @file config.cuh
 * @brief Configuration and simulation parameters for G-equation Level-Set solver
 *
 * This header defines all configurable parameters for the simulation including:
 * - Grid dimensions and resolution
 * - Physical parameters (flame speed, velocity field)
 * - Numerical scheme parameters
 * - Reinitialization settings
 */

#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <cuda_runtime.h>
#include <cmath>

//=============================================================================
// Grid Configuration
//=============================================================================

// Number of interior grid points (excluding ghost cells)
constexpr int NX = 401;
constexpr int NY = 401;

// Number of ghost cells on each side (required for WENO-5)
constexpr int NGHOST = 3;

// Total grid dimensions including ghost cells
constexpr int NX_TOTAL = NX + 2 * NGHOST;
constexpr int NY_TOTAL = NY + 2 * NGHOST;

// Physical domain size
constexpr double DOMAIN_X_MIN = 0.0;
constexpr double DOMAIN_X_MAX = 1.0;
constexpr double DOMAIN_Y_MIN = 0.0;
constexpr double DOMAIN_Y_MAX = 1.0;

// Grid spacing
constexpr double DX = (DOMAIN_X_MAX - DOMAIN_X_MIN) / (NX - 1);
constexpr double DY = (DOMAIN_Y_MAX - DOMAIN_Y_MIN) / (NY - 1);

//=============================================================================
// Physical Parameters
//=============================================================================

// Laminar flame speed (S_L = 0 for pure advection test)
constexpr double S_L = 0.0;

// Constant advection velocity field
constexpr double U_CONST = 1.0;  // x-direction velocity
constexpr double V_CONST = 0.0;  // y-direction velocity

//=============================================================================
// Time Integration Parameters
//=============================================================================

// CFL number for time step calculation
constexpr double CFL = 0.2;

// Total simulation time
constexpr double T_FINAL = 6.283185;

// Maximum number of time steps (safety limit)
constexpr int MAX_STEPS = 1000000;

// Output frequency (every N steps)
constexpr int OUTPUT_INTERVAL = 10;

//=============================================================================
// Reinitialization Parameters (HCR-2 Method)
//=============================================================================

// Enable/disable reinitialization
constexpr bool ENABLE_REINIT = false;

// Reinitialization frequency (every N time steps)
constexpr int REINIT_INTERVAL = 10;

// Number of pseudo-time iterations for reinitialization
constexpr int REINIT_ITERATIONS = 2;

// Pseudo-time step for reinitialization: Δτ = Δx/4 (Hartmann et al. 2010, p.5)
constexpr double REINIT_DTAU_FACTOR = 0.25;

// Weighting factor for forcing term (beta = 0.5 as specified)
constexpr double REINIT_BETA = 0.5;

// Small epsilon for numerical stability
constexpr double EPSILON = 1.0e-10;

// Band width for interface detection (in terms of grid cells)
constexpr double INTERFACE_BAND = 2.0;

//=============================================================================
// CUDA Configuration
//=============================================================================

// Thread block dimensions
constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;

// Calculate grid dimensions for kernel launches
inline dim3 getGridDim() {
    return dim3((NX_TOTAL + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (NY_TOTAL + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
}

inline dim3 getBlockDim() {
    return dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);
}

//=============================================================================
// Simulation Parameters Structure
//=============================================================================

/**
 * @struct SimParams
 * @brief Structure holding all simulation parameters for device access
 */
struct SimParams {
    // Grid parameters
    int nx, ny;
    int nx_total, ny_total;
    int nghost;
    double dx, dy;
    double x_min, x_max;
    double y_min, y_max;

    // Physical parameters
    double s_l;           // Laminar flame speed
    double u_const;       // Constant x-velocity
    double v_const;       // Constant y-velocity

    // Time parameters
    double dt;            // Current time step
    double t_final;       // Final simulation time
    double cfl;           // CFL number

    // Reinitialization parameters
    bool enable_reinit;
    int reinit_interval;
    int reinit_iterations;
    double reinit_dtau;
    double reinit_beta;

    // Numerical parameters
    double epsilon;
    double interface_band;
};

/**
 * @brief Initialize simulation parameters with default values
 */
inline SimParams initDefaultParams() {
    SimParams params;

    // Grid
    params.nx = NX;
    params.ny = NY;
    params.nx_total = NX_TOTAL;
    params.ny_total = NY_TOTAL;
    params.nghost = NGHOST;
    params.dx = DX;
    params.dy = DY;
    params.x_min = DOMAIN_X_MIN;
    params.x_max = DOMAIN_X_MAX;
    params.y_min = DOMAIN_Y_MIN;
    params.y_max = DOMAIN_Y_MAX;

    // Physical
    params.s_l = S_L;
    params.u_const = U_CONST;
    params.v_const = V_CONST;

    // Time
    params.dt = 0.0;  // Will be calculated based on CFL
    params.t_final = T_FINAL;
    params.cfl = CFL;

    // Reinitialization
    params.enable_reinit = ENABLE_REINIT;
    params.reinit_interval = REINIT_INTERVAL;
    params.reinit_iterations = REINIT_ITERATIONS;
    params.reinit_dtau = REINIT_DTAU_FACTOR * fmin(DX, DY);
    params.reinit_beta = REINIT_BETA;

    // Numerical
    params.epsilon = EPSILON;
    params.interface_band = INTERFACE_BAND;

    return params;
}

/**
 * @brief Calculate time step based on CFL condition
 * @param params Simulation parameters
 * @return Computed time step
 */
inline double calculateTimeStep(const SimParams& params) {
    // Maximum wave speed: |u| + |v| + S_L
    double max_speed = fabs(params.u_const) + fabs(params.v_const) + params.s_l;

    if (max_speed < params.epsilon) {
        max_speed = params.epsilon;  // Avoid division by zero
    }

    // CFL condition: dt <= CFL * min(dx, dy) / max_speed
    double dt = params.cfl * fmin(params.dx, params.dy) / max_speed;

    return dt;
}

//=============================================================================
// CUDA Error Checking Macro
//=============================================================================

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

//=============================================================================
// Array Index Macros
//=============================================================================

// Convert 2D indices to 1D array index (row-major order)
#define IDX(i, j) ((j) * NX_TOTAL + (i))

// Device-compatible index calculation
__host__ __device__ inline int idx(int i, int j, int nx_total) {
    return j * nx_total + i;
}

// Convert physical coordinates to grid indices
__host__ __device__ inline void coordToIndex(double x, double y,
                                              double x_min, double y_min,
                                              double dx, double dy, int nghost,
                                              int& i, int& j) {
    i = static_cast<int>((x - x_min) / dx) + nghost;
    j = static_cast<int>((y - y_min) / dy) + nghost;
}

// Convert grid indices to physical coordinates
__host__ __device__ inline void indexToCoord(int i, int j,
                                              double x_min, double y_min,
                                              double dx, double dy, int nghost,
                                              double& x, double& y) {
    x = x_min + (i - nghost) * dx;
    y = y_min + (j - nghost) * dy;
}

#endif // CONFIG_CUH
