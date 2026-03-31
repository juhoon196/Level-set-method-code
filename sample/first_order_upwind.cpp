#include "first_order_upwind.hpp"
#include "config.hpp"
#include <cmath>
#include <omp.h>

void compute_first_order_upwind(
    std::vector<std::vector<double>>& G,
    const std::vector<std::vector<double>>& u,
    const std::vector<std::vector<double>>& v,
    double dx,
    double dy,
    double dt,
    double S_L
) {
    int ny = G.size();
    int nx = G[0].size();

    std::vector<std::vector<double>> G_new = G;

    double Gx_mns, Gx_pls, Gy_mns, Gy_pls;
    double dG_dx, dG_dy;
    double a_pls, a_mns, b_pls, b_mns;

    const double EPSILON = 1e-10;

    #pragma omp parallel for collapse(2)
    for (int j = 3; j <= ny-3; ++j) {
        for (int i = 3; i <= nx-3; ++i) {
            // (dudx, dvdy 변수는 사용되지 않으므로 제거 가능)

            // Gx_mns, Gx_pls는 dx로 나눔 (X방향)
            Gx_mns = (G[j][i] - G[j][i-1]) / dx;
            Gx_pls = (G[j][i+1] - G[j][i]) / dx;

            // Gy_mns, Gy_pls는 dy로 나눔 (Y방향)
            Gy_mns = (G[j][i] - G[j-1][i]) / dy;
            Gy_pls = (G[j+1][i] - G[j][i]) / dy;

            dG_dx = 0.5 * (Gx_mns + Gx_pls);
            dG_dy = 0.5 * (Gy_mns + Gy_pls);

            double mag_gradG_sq = dG_dx * dG_dx + dG_dy * dG_dy;
            double u_first, v_first;

            if (mag_gradG_sq > EPSILON * EPSILON) { 
                double mag_gradG = std::sqrt(mag_gradG_sq);
                u_first = u[j][i] - S_L * dG_dx / mag_gradG;
                v_first = v[j][i] - S_L * dG_dy / mag_gradG;
            } else {
                
                u_first = u[j][i];
                v_first = v[j][i];
            }

            a_pls = std::max(u_first, 0.0);
            a_mns = std::min(u_first, 0.0);
            b_pls = std::max(v_first, 0.0);
            b_mns = std::min(v_first, 0.0);

            G_new[j][i] = G[j][i] - dt * (a_pls*Gx_mns + a_mns*Gx_pls + b_pls*Gy_mns + b_mns*Gy_pls);
        }
    }


    #pragma omp parallel for collapse(2) 
    for (int j = 3; j <= ny-3; ++j) {
        for (int i = 3; i <= nx-3; ++i) {
            G[j][i] = G_new[j][i];
        }
    }
}
