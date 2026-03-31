#pragma once

const int num_threads = 16;
const int NX = 800;
const int NY = 800;
const double LX = 0.02;
const double LY = 0.02;
const double DX = LX / (NX - 1);
const double DY = LY / (NY - 1);
const double FINAL_TIME = 0.16;
const double DT = 0.0000025;
const double SL = 0.4;  // 0.4

const double u0 = 0.0;  // 0.8
const double v0 = 0.8;
const double epsilon_u = 0.05;
const double epsilon_v = 0.05;
const int reinit_interval = 1;
const int reinit_itr = 10;
const int save_interval = 20;
