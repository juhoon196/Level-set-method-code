/**
 * @file config.h
 * @brief Configuration and simulation parameters for G-equation Level-Set solver (MPI 3D version)
 *
 * This header defines all configurable parameters for the 3D simulation including:
 * - Grid dimensions and resolution
 * - Physical parameters (flame speed, velocity field)
 * - Numerical scheme parameters
 * - Reinitialization settings
 * - MPI domain decomposition parameters (1D decomposition in Y direction)
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
constexpr int NX = 64;
constexpr int NY = 64;
constexpr int NZ = 64;

// Number of ghost cells on each side (required for WENO-5)
constexpr int NGHOST = 3;

// Total grid dimensions including ghost cells - GLOBAL
constexpr int NX_TOTAL = NX + 2 * NGHOST;
constexpr int NY_TOTAL = NY + 2 * NGHOST;
constexpr int NZ_TOTAL = NZ + 2 * NGHOST;

// Physical domain size
constexpr double DOMAIN_X_MIN = 0.0;
constexpr double DOMAIN_X_MAX = 1.0;
constexpr double DOMAIN_Y_MIN = 0.0;
constexpr double DOMAIN_Y_MAX = 1.0;
constexpr double DOMAIN_Z_MIN = 0.0;
constexpr double DOMAIN_Z_MAX = 1.0;

// Grid spacing
constexpr double DX = (DOMAIN_X_MAX - DOMAIN_X_MIN) / (NX - 1);
constexpr double DY = (DOMAIN_Y_MAX - DOMAIN_Y_MIN) / (NY - 1);
constexpr double DZ = (DOMAIN_Z_MAX - DOMAIN_Z_MIN) / (NZ - 1);

//=============================================================================
// Physical Parameters
//=============================================================================

// Laminar flame speed (S_L = 0 for pure advection test)
constexpr double S_L = 0.0;

// Constant advection velocity field (not used for deformation test)
constexpr double U_CONST = 0.0;  // x-direction velocity
constexpr double V_CONST = 0.0;  // y-direction velocity
constexpr double W_CONST = 0.0;  // z-direction velocity

//=============================================================================
// Time Integration Parameters
//=============================================================================

// Time step (DT)
// If DT = 0.0: automatically calculated based on CFL condition
// If DT > 0.0: use this fixed time step value
constexpr double DT = 0.001;

// CFL number for time step calculation (used only when DT = 0.0)
constexpr double CFL = 0.2;

// Total simulation time (T=1.5 for deformation test)
constexpr double T_FINAL = 1.5;

// Maximum number of time steps (safety limit)
constexpr int MAX_STEPS = 1000000;

// Output frequency (every N steps)
constexpr int OUTPUT_INTERVAL = 50;

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
    int nx, ny, nz;
    int nx_total, ny_total, nz_total;
    int nghost;
    double dx, dy, dz;
    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;

    // MPI domain decomposition parameters (1D in Y direction)
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
    double w_const;       // Constant z-velocity

    // Time parameters
    double dt;            // Current time step
    double t_final;       // Final simulation time
    double cfl;           // CFL number
    double current_time;  // Current simulation time (for time-dependent velocity)

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
    params.nz = NZ;
    params.nx_total = NX_TOTAL;
    params.ny_total = NY_TOTAL;
    params.nz_total = NZ_TOTAL;
    params.nghost = NGHOST;
    params.dx = DX;
    params.dy = DY;
    params.dz = DZ;
    params.x_min = DOMAIN_X_MIN;
    params.x_max = DOMAIN_X_MAX;
    params.y_min = DOMAIN_Y_MIN;
    params.y_max = DOMAIN_Y_MAX;
    params.z_min = DOMAIN_Z_MIN;
    params.z_max = DOMAIN_Z_MAX;

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
    params.w_const = W_CONST;

    // Time
    params.dt = DT;  // If 0.0: CFL-based auto calculation, >0.0: use this fixed value
    params.t_final = T_FINAL;
    params.cfl = CFL;
    params.current_time = 0.0;

    // Reinitialization
    params.enable_reinit = ENABLE_REINIT;
    params.reinit_interval = REINIT_INTERVAL;
    params.reinit_iterations = REINIT_ITERATIONS;
    params.reinit_dtau = REINIT_DTAU_FACTOR * fmin(DX, fmin(DY, DZ));
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

    // Set neighbors (periodic in Y: wrap around)
    params.neighbor_below = (rank > 0) ? rank - 1 : num_procs - 1;
    params.neighbor_above = (rank < num_procs - 1) ? rank + 1 : 0;
}

/**
 * @brief Calculate time step based on CFL condition
 * @param params Simulation parameters
 * @return Computed time step
 */
inline double calculateTimeStep(const SimParams& params) {
    // If params.dt is set to a specific value (by user), use it.
    // Otherwise (if 0.0), calculate based on CFL.
    if (params.dt > 0.0) {
        return params.dt;
    }

    // For deformation test, max velocity is about 2.0
    double max_speed = fabs(params.u_const) + fabs(params.v_const)
                     + fabs(params.w_const) + params.s_l;

    // If using deformation velocity field, estimate max velocity
    if (max_speed < params.epsilon) {
        max_speed = 2.0;  // Max velocity from deformation field
    }

    // CFL condition: dt <= CFL * min(dx, dy, dz) / max_speed
    double dmin = fmin(params.dx, fmin(params.dy, params.dz));
    double dt = params.cfl * dmin / max_speed;

    return dt;
}

//=============================================================================
// Array Index Macros (3D)
//=============================================================================

// Convert 3D indices to 1D array index (row-major order)
// Memory layout: x is fastest, then y, then z (i + j*nx_total + k*nx_total*ny_local_total)
#define IDX3(i, j, k, nx_total, ny_total) ((k) * (nx_total) * (ny_total) + (j) * (nx_total) + (i))

// Standard index function for local 3D arrays
inline int idx3(int i, int j, int k, int nx_total, int ny_total) {
    return k * nx_total * ny_total + j * nx_total + i;
}

// Convert LOCAL grid indices to physical coordinates (3D)
inline void indexToCoord3D(int i, int j, int k,
                           double x_min, double y_min, double z_min,
                           double dx, double dy, double dz, int nghost,
                           int y_start,
                           double& x, double& y, double& z) {
    x = x_min + (i - nghost) * dx;
    int global_j = y_start + (j - nghost);
    y = y_min + global_j * dy;
    z = z_min + (k - nghost) * dz;
}

// Backward compatible version (without y_start offset)
inline void indexToCoord3D(int i, int j, int k,
                           double x_min, double y_min, double z_min,
                           double dx, double dy, double dz, int nghost,
                           double& x, double& y, double& z) {
    x = x_min + (i - nghost) * dx;
    y = y_min + (j - nghost) * dy;
    z = z_min + (k - nghost) * dz;
}

#endif // CONFIG_H
