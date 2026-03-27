/**
 * @file config.h
 * @brief Configuration and simulation parameters for G-equation Level-Set solver (MPI version)
 *
 * This header defines all configurable parameters for the simulation including:
 * - Grid dimensions and resolution
 * - Physical parameters (flame speed, velocity field)
 * - Numerical scheme parameters
 * - Reinitialization settings
 * - MPI domain decomposition parameters
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

//=============================================================================
// Grid Configuration
//=============================================================================

// Number of interior grid points (excluding ghost cells) - GLOBAL
constexpr int NX = 1601;
constexpr int NY = 1601;

// Number of ghost cells on each side (required for WENO-5)
constexpr int NGHOST = 3;

// Total grid dimensions including ghost cells - GLOBAL
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
// MPI Error Checking Macro
//=============================================================================

#define MPI_CHECK(call)                                                       \
    do {                                                                      \
        int err = call;                                                       \
        if (err != MPI_SUCCESS) {                                             \
            char error_string[MPI_MAX_ERROR_STRING];                          \
            int length;                                                       \
            MPI_Error_string(err, error_string, &length);                     \
            fprintf(stderr, "MPI error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, error_string);                        \
            MPI_Abort(MPI_COMM_WORLD, err);                                   \
        }                                                                     \
    } while (0)

//=============================================================================
// Simulation Parameters Structure
//=============================================================================

/**
 * @struct SimParams
 * @brief Structure holding all simulation parameters including MPI info
 */
struct SimParams {
    // Grid parameters - GLOBAL
    int nx, ny;
    int nx_total, ny_total;
    int nghost;
    double dx, dy;
    double x_min, x_max;
    double y_min, y_max;

    // MPI domain decomposition parameters
    int rank;           // Current process rank
    int num_procs;      // Total number of processes
    int local_ny;       // Number of interior rows for this process
    int local_ny_total; // Total rows including ghost cells
    int y_start;        // Global y-index start for this process (interior only)
    int neighbor_below; // Rank of neighbor below (or MPI_PROC_NULL)
    int neighbor_above; // Rank of neighbor above (or MPI_PROC_NULL)

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

    // Grid - Global
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

    // MPI - will be set later
    params.rank = 0;
    params.num_procs = 1;
    params.local_ny = NY;
    params.local_ny_total = NY_TOTAL;
    params.y_start = 0;
    params.neighbor_below = MPI_PROC_NULL;
    params.neighbor_above = MPI_PROC_NULL;

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
 * @brief Setup MPI domain decomposition (1D decomposition in Y direction)
 * @param params Simulation parameters to update
 * @param rank MPI rank
 * @param num_procs Total number of MPI processes
 */
inline void setupMPIDomainDecomposition(SimParams& params, int rank, int num_procs) {
    params.rank = rank;
    params.num_procs = num_procs;

    // Divide NY rows among processes
    int base_rows = params.ny / num_procs;
    int remainder = params.ny % num_procs;

    // First 'remainder' processes get one extra row
    if (rank < remainder) {
        params.local_ny = base_rows + 1;
        params.y_start = rank * (base_rows + 1);
    } else {
        params.local_ny = base_rows;
        params.y_start = remainder * (base_rows + 1) + (rank - remainder) * base_rows;
    }

    // Total local size including ghost cells
    params.local_ny_total = params.local_ny + 2 * params.nghost;

    // Set neighbors
    params.neighbor_below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    params.neighbor_above = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;
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
// Array Index Macros
//=============================================================================

// Convert 2D indices to 1D array index (row-major order)
// For local arrays: j is local index [0, local_ny_total)
#define IDX(i, j, nx_total) ((j) * (nx_total) + (i))

// Standard index function for local arrays
inline int idx(int i, int j, int nx_total) {
    return j * nx_total + i;
}

// Convert LOCAL grid indices to physical coordinates
// i: x-index in [0, nx_total), includes ghost cells
// j: LOCAL y-index in [0, local_ny_total), includes ghost cells
inline void indexToCoord(int i, int j,
                         double x_min, double y_min,
                         double dx, double dy, int nghost,
                         int y_start,  // global y-start (interior only)
                         double& x, double& y) {
    // x-coordinate: straightforward
    x = x_min + (i - nghost) * dx;
    
    // y-coordinate: must account for global offset
    // j is local index, j - nghost is local interior index
    // Global interior index = y_start + (j - nghost)
    int global_j = y_start + (j - nghost);
    y = y_min + global_j * dy;
}

// Convert grid indices to physical coordinates (backward compatible version)
inline void indexToCoord(int i, int j,
                         double x_min, double y_min,
                         double dx, double dy, int nghost,
                         double& x, double& y) {
    x = x_min + (i - nghost) * dx;
    y = y_min + (j - nghost) * dy;
}

#endif // CONFIG_H
