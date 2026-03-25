#include "config.hpp"
#include "field.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

constexpr double EPSILON = 1e-6;

// --- WENO5 helper: smoothness indicator ---
inline double beta(double f0, double f1, double f2) {
    return (13.0/12.0) * std::pow(f0 - 2.0*f1 + f2, 2)
         + (1.0/4.0) * std::pow(f0 - 4.0*f1 + 3.0*f2, 2);
}

// --- WENO5 left-biased 1D stencil ---
inline double weno5_left(double v0, double v1, double v2, double v3, double v4) {
    double b0 = beta(v0, v1, v2);
    double b1 = beta(v1, v2, v3);
    double b2 = beta(v2, v3, v4);
    double g0 = 0.1, g1 = 0.6, g2 = 0.3;
    double a0 = g0 / ((EPSILON + b0)*(EPSILON + b0));
    double a1 = g1 / ((EPSILON + b1)*(EPSILON + b1));
    double a2 = g2 / ((EPSILON + b2)*(EPSILON + b2));
    double sum = a0 + a1 + a2;
    double w0 = a0 / sum;
    double w1 = a1 / sum;
    double w2 = a2 / sum;
    return w0 * (2.0*v0 - 7.0*v1 + 11.0*v2)/6.0
         + w1 * (-v1 + 5.0*v2 + 2.0*v3)/6.0
         + w2 * (2.0*v2 + 5.0*v3 - v4)/6.0;
}

// --- WENO5 right-biased 1D stencil ---
inline double weno5_right(double v0, double v1, double v2, double v3, double v4) {
    double b0 = beta(v4, v3, v2);
    double b1 = beta(v3, v2, v1);
    double b2 = beta(v2, v1, v0);
    double g0 = 0.1, g1 = 0.6, g2 = 0.3;
    double a0 = g0 / ((EPSILON + b0)*(EPSILON + b0));
    double a1 = g1 / ((EPSILON + b1)*(EPSILON + b1));
    double a2 = g2 / ((EPSILON + b2)*(EPSILON + b2));
    double sum = a0 + a1 + a2;
    double w0 = a0 / sum;
    double w1 = a1 / sum;
    double w2 = a2 / sum;
    return w0 * (2.0*v4 - 7.0*v3 + 11.0*v2)/6.0
         + w1 * (-v3 + 5.0*v2 + 2.0*v1)/6.0
         + w2 * (2.0*v2 + 5.0*v1 - v0)/6.0;
}

// --- X-방향 WENO5 도함수 (left/right biased) ---
void weno5_deriv_x(const std::vector<std::vector<double>>& G,
    std::vector<std::vector<double>>& dG_L,
    std::vector<std::vector<double>>& dG_R, double dx) {
    int ny = G.size();        // = NY+6
    int nx = G[0].size();     // = NX+6
    dG_L.assign(ny, std::vector<double>(nx, 0.0));
    dG_R.assign(ny, std::vector<double>(nx, 0.0));
    for (int j = 3; j <= ny-4; ++j) {
        for (int i = 3; i <= nx-4; ++i) {
        dG_L[j][i] = (weno5_left(G[j][i-2], G[j][i-1], G[j][i], G[j][i+1], G[j][i+2])
                - weno5_left(G[j][i-3], G[j][i-2], G[j][i-1], G[j][i], G[j][i+1])) / dx;
        dG_R[j][i] = (weno5_right(G[j][i+3], G[j][i+2], G[j][i+1], G[j][i], G[j][i-1])
                - weno5_right(G[j][i+2], G[j][i+1], G[j][i], G[j][i-1], G[j][i-2])) / dx;
        }
    }
}


// --- Y-방향 WENO5 도함수 (left/right biased) ---
void weno5_deriv_y(const std::vector<std::vector<double>>& G,
    std::vector<std::vector<double>>& dG_L,
    std::vector<std::vector<double>>& dG_R, double dy) {
    int ny = G.size();        // = NY+6
    int nx = G[0].size();     // = NX+6
    dG_L.assign(ny, std::vector<double>(nx, 0.0));
    dG_R.assign(ny, std::vector<double>(nx, 0.0));
    for (int j = 3; j <= ny-4; ++j) {
        for (int i = 3; i <= nx-4; ++i) {
        dG_L[j][i] = (weno5_left(G[j-2][i], G[j-1][i], G[j][i], G[j+1][i], G[j+2][i])
                - weno5_left(G[j-3][i], G[j-2][i], G[j-1][i], G[j][i], G[j+1][i])) / dy;
        dG_R[j][i] = (weno5_right(G[j+3][i], G[j+2][i], G[j+1][i], G[j][i], G[j-1][i])
                - weno5_right(G[j+2][i], G[j+1][i], G[j][i], G[j-1][i], G[j-2][i])) / dy;
        }
    }
}



void compute_geqn_rhs(
    const std::vector<std::vector<double>>& G,
    const std::vector<std::vector<double>>& u,
    const std::vector<std::vector<double>>& v,
    std::vector<std::vector<double>>& rhs,
    double dx, double dy, double S_L)
{
    int ny = G.size() - 6;
    int nx = G[0].size() - 6;

    // 左右偏差分 (WENO5)
    std::vector<std::vector<double>> dGdx_L(G.size(), std::vector<double>(G[0].size()));
    std::vector<std::vector<double>> dGdx_R = dGdx_L;
    std::vector<std::vector<double>> dGdy_L = dGdx_L;
    std::vector<std::vector<double>> dGdy_R = dGdx_L;
    weno5_deriv_x(G, dGdx_L, dGdx_R, dx);
    weno5_deriv_y(G, dGdy_L, dGdy_R, dy);

    rhs.assign(G.size(), std::vector<double>(G[0].size(), 0.0));

    #pragma omp parallel for collapse(2)
    for (int j = 3; j <= ny+2; ++j) {
        for (int i = 3; i <= nx+2; ++i) {
            // —–– 평균도함수 (central part)
            double Gdxm = dGdx_L[j][i], Gdxp = dGdx_R[j][i];
            double Gdym = dGdy_L[j][i], Gdyp = dGdy_R[j][i];
            double Gdx_mean = 0.5*(Gdxm + Gdxp);
            double Gdy_mean = 0.5*(Gdym + Gdyp);

            // 이류＋전파
            double transport   = u[j][i]*Gdx_mean + v[j][i]*Gdy_mean;
            double propagation = S_L * std::sqrt(Gdx_mean*Gdx_mean + Gdy_mean*Gdy_mean);

            // —— 국소 Lax–Friedrichs 속도 α 계산
            // x−방향
            double s_xm = u[j][i] - S_L * Gdxm/std::sqrt(Gdxm*Gdxm + Gdym*Gdym + 1e-14);
            double s_xp = u[j][i] - S_L * Gdxp/std::sqrt(Gdxp*Gdxp + Gdyp*Gdyp + 1e-14);
            double alpha_x = std::max(std::abs(s_xm), std::abs(s_xp));
            // y−방향
            double s_ym = v[j][i] - S_L * Gdym/std::sqrt(Gdxm*Gdxm + Gdym*Gdym + 1e-14);
            double s_yp = v[j][i] - S_L * Gdyp/std::sqrt(Gdxp*Gdxp + Gdyp*Gdyp + 1e-14);
            double alpha_y = std::max(std::abs(s_ym), std::abs(s_yp));
            double alpha = std::max(alpha_x, alpha_y);

            // —— 인공 점성 (dissipation)
            double diss_x = alpha * (Gdxp - Gdxm) * 0.5;
            double diss_y = alpha * (Gdyp - Gdym) * 0.5;

            // Hamiltonian
            double H = transport - propagation - (diss_x + diss_y);

            rhs[j][i] = -H;
        }
    }
}


void RK3_step(
    Field2D& G_field,
    const Field2D& u_field,
    const Field2D& v_field,
    double dx, double dy, double dt, double S_L,
    const Field2D& G_initial
) {
    auto& G = G_field.data;
    const auto& u = u_field.data;
    const auto& v = v_field.data;
    int ny = G.size() - 6;
    int nx = G[0].size() - 6;

    Field2D G1_field(G_field.nx, G_field.ny);
    Field2D G2_field(G_field.nx, G_field.ny);

    std::vector<std::vector<double>> rhs(G.size(), std::vector<double>(G[0].size(), 0.0));

    // Stage 1
    compute_geqn_rhs(G, u, v, rhs, dx, dy, S_L);
    for (int j = 3; j <= ny+2; ++j)
        for (int i = 3; i <= nx+2; ++i)
            G1_field.data[j][i] = G[j][i] + dt * rhs[j][i];
    G1_field.apply_anchored_bc(G_initial);
    G1_field.apply_periodic();
    // Stage 2
    compute_geqn_rhs(G1_field.data, u, v, rhs, dx, dy, S_L);
    for (int j = 3; j <= ny+2; ++j)
        for (int i = 3; i <= nx+2; ++i)
            G2_field.data[j][i] = 0.75 * G[j][i] + 0.25 * (G1_field.data[j][i] + dt * rhs[j][i]);
    G2_field.apply_anchored_bc(G_initial);
    G2_field.apply_periodic();

    // Stage 3
    compute_geqn_rhs(G2_field.data, u, v, rhs, dx, dy, S_L);
    for (int j = 3; j <= ny+2; ++j)
        for (int i = 3; i <= nx+2; ++i)
            G[j][i] = (1.0/3.0) * G[j][i] + (2.0/3.0) * (G2_field.data[j][i] + dt * rhs[j][i]);
    G_field.apply_anchored_bc(G_initial);
    G_field.apply_periodic();
}


