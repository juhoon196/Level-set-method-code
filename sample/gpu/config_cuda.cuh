#pragma once

// Grid configuration
constexpr int NX = 800;
constexpr int NY = 800;
constexpr int GHOST = 3;
constexpr int NX_TOTAL = NX + 2 * GHOST;
constexpr int NY_TOTAL = NY + 2 * GHOST;

// Domain size
constexpr double LX = 0.02;
constexpr double LY = 0.02;
constexpr double DX = LX / (NX - 1);
constexpr double DY = LY / (NY - 1);

// Time integration
constexpr double FINAL_TIME = 0.16;
constexpr double DT = 0.0000025;

// Physical parameters
constexpr double SL = 0.4;
constexpr double u0 = 0.0;
constexpr double v0 = 0.8;
constexpr double epsilon_u = 0.05;
constexpr double epsilon_v = 0.05;

// Reinitialization (optional)
constexpr int reinit_interval = 1;
constexpr int reinit_itr = 10;

// Output
constexpr int save_interval = 20;

// CUDA configuration
constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;

// WENO epsilon
constexpr double WENO_EPSILON = 1e-6;
