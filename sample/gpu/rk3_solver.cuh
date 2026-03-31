#pragma once
#include "config_cuda.cuh"
#include "field_cuda.cuh"
#include "weno5_kernels.cuh"

// RK3 stage update kernels
__global__ void kernel_rk3_stage1(
    const double* G,
    const double* rhs,
    double* G1,
    double dt
);

__global__ void kernel_rk3_stage2(
    const double* G,
    const double* G1,
    const double* rhs,
    double* G2,
    double dt
);

__global__ void kernel_rk3_stage3(
    double* G,
    const double* G2,
    const double* rhs,
    double dt
);

// Velocity update kernel
__global__ void kernel_update_velocity(
    double* u,
    double* v,
    double t,
    int freq
);

// Complete RK3 step wrapper
void RK3_step_gpu(
    double* d_G,
    double* d_G1,
    double* d_G2,
    const double* d_u,
    const double* d_v,
    double* d_rhs,
    const double* d_G_initial,
    double dx,
    double dy,
    double dt,
    double S_L
);
