#pragma once
#include "config_cuda.cuh"
#include "field_cuda.cuh"

// WENO5 derivative kernels
__global__ void kernel_weno5_deriv_x(const double* G, double* dG_L, double* dG_R, double dx);
__global__ void kernel_weno5_deriv_y(const double* G, double* dG_L, double* dG_R, double dy);

// G-equation RHS computation
__global__ void kernel_compute_geqn_rhs(
    const double* G,
    const double* u,
    const double* v,
    const double* dGdx_L,
    const double* dGdx_R,
    const double* dGdy_L,
    const double* dGdy_R,
    double* rhs,
    double S_L
);

// Combined WENO5 + RHS computation (more efficient)
__global__ void kernel_compute_rhs_weno5(
    const double* G,
    const double* u,
    const double* v,
    double* rhs,
    double dx,
    double dy,
    double S_L
);

// Wrapper functions
void compute_weno5_deriv_x_gpu(const double* d_G, double* d_dG_L, double* d_dG_R, double dx);
void compute_weno5_deriv_y_gpu(const double* d_G, double* d_dG_L, double* d_dG_R, double dy);
void compute_geqn_rhs_gpu(
    const double* d_G,
    const double* d_u,
    const double* d_v,
    double* d_rhs,
    double dx,
    double dy,
    double S_L
);
