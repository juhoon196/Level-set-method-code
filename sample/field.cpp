#include "field.hpp"
#include <cmath>
#include <algorithm>

Field2D::Field2D(int nx_, int ny_) : nx(nx_), ny(ny_) {
    // ghost cell 3줄씩 → (0~nx+5, 0~ny+5)
    data.resize(ny + 6, std::vector<double>(nx + 6, 0.0));
}

void Field2D::fill(double value) {
    for (auto& row : data)
        std::fill(row.begin(), row.end(), value);
}

// 좌우, 상하, 모서리 모두 periodic ghost cell 복사
void Field2D::apply_periodic() {
    int nx_total = nx + 6, ny_total = ny + 6;
    // 좌우
    for (int j = 0; j < ny_total; ++j) {
        for (int k = 0; k < 3; ++k) {
            data[j][k]          = data[j][nx + k];
            data[j][nx + 3 + k] = data[j][3 + k];
        }
    }
    // 상하
    for (int i = 0; i < nx_total; ++i) {
        for (int k = 0; k < 3; ++k) {
            data[k][i]          = data[ny + k][i];
            data[ny + 3 + k][i] = data[3 + k][i];
        }
    }
    // 모서리(일관성 유지, 필요한 경우만)
    // 실제로는 위의 반복문이 모두 처리함
}

// 하단(anchoring), 나머지 선형외삽
void Field2D::apply_anchored_bc(const Field2D& initial) {
    int nx_total = nx + 6, ny_total = ny + 6;
    // 하단: anchored (initial 값 복사)
    for (int i = 0; i < nx_total; ++i) {
        data[3][i] = initial.data[3][i]; // anchoring
        data[2][i] = 2.0 * data[3][i] - data[4][i];
        data[1][i] = 2.0 * data[2][i] - data[3][i];
        data[0][i] = 2.0 * data[1][i] - data[2][i];
    }
    // 상단: 선형외삽
    for (int i = 0; i < nx_total; ++i) {
        data[ny + 2][i] = 2.0 * data[ny + 1][i] - data[ny][i];
        data[ny + 3][i] = 2.0 * data[ny + 2][i] - data[ny + 1][i];
        data[ny + 4][i] = 2.0 * data[ny + 3][i] - data[ny + 2][i];
        data[ny + 5][i] = 2.0 * data[ny + 4][i] - data[ny + 3][i];
    }
    // 좌/우: 선형외삽
    for (int j = 0; j < ny_total; ++j) {
        data[j][2] = 2.0 * data[j][3] - data[j][4];
        data[j][1] = 2.0 * data[j][2] - data[j][3];
        data[j][0] = 2.0 * data[j][1] - data[j][2];

        data[j][nx + 3] = 2.0 * data[j][nx + 2] - data[j][nx + 1];
        data[j][nx + 4] = 2.0 * data[j][nx + 3] - data[j][nx + 2];
        data[j][nx + 5] = 2.0 * data[j][nx + 4] - data[j][nx + 3];
    }
}

// Level Set reinitialization (Osher-Sethian/fast marching/ENORM 등 구현 가능, 여기선 1st order Osher-Sethian)
void Field2D::reinitialize(const Field2D& initial, double dx, double dy, int num_iters) {
    int nx_total = nx + 6, ny_total = ny + 6;
    double dtau = 0.2 * std::min(dx, dy);

    std::vector<std::vector<double>> phi = data;

    for (int iter = 0; iter < num_iters; ++iter) {
        std::vector<std::vector<double>> rhs(ny_total, std::vector<double>(nx_total, 0.0));

        #pragma omp parallel for collapse(2)
        for (int j = 3; j <= ny + 2; ++j) {
            for (int i = 3; i <= nx + 2; ++i) {
                double s = initial.data[j][i] / std::sqrt(initial.data[j][i] * initial.data[j][i] + dx * dx);
                
                // 1차 차분 (forward/backward)
                double dGdx_pos = (phi[j][i] - phi[j][i-1]) / dx;
                double dGdx_neg = (phi[j][i+1] - phi[j][i]) / dx;
                double dGdy_pos = (phi[j][i] - phi[j-1][i]) / dy;
                double dGdy_neg = (phi[j+1][i] - phi[j][i]) / dy;

                double grad;
                if (s > 0.0) {
                    double px = std::max(std::max(dGdx_pos, 0.0), -std::min(dGdx_neg, 0.0));
                    double py = std::max(std::max(dGdy_pos, 0.0), -std::min(dGdy_neg, 0.0));
                    grad = std::sqrt(px * px + py * py);
                } else {
                    double nx = std::max(std::max(-dGdx_pos, 0.0), std::min(dGdx_neg, 0.0));
                    double ny = std::max(std::max(-dGdy_pos, 0.0), std::min(dGdy_neg, 0.0));
                    grad = std::sqrt(nx * nx + ny * ny);
                }
                rhs[j][i] = -s * (grad - 1.0);
            }
        }
        // Forward Euler update
        #pragma omp parallel for collapse(2)
        for (int j = 3; j <= ny + 2; ++j) {
            for (int i = 3; i <= nx + 2; ++i) {
                phi[j][i] += dtau * rhs[j][i];
            }
        }
        // Boundary condition (anchoring/ghost)
        for (int i = 0; i < nx_total; ++i) {
            phi[3][i] = initial.data[3][i];
            phi[2][i] = 2.0 * phi[3][i] - phi[4][i];
            phi[1][i] = 2.0 * phi[2][i] - phi[3][i];
            phi[0][i] = 2.0 * phi[1][i] - phi[2][i];
        }
        for (int i = 0; i < nx_total; ++i) {
            phi[ny + 2][i] = 2.0 * phi[ny + 1][i] - phi[ny][i];
            phi[ny + 3][i] = 2.0 * phi[ny + 2][i] - phi[ny + 1][i];
            phi[ny + 4][i] = 2.0 * phi[ny + 3][i] - phi[ny + 2][i];
            phi[ny + 5][i] = 2.0 * phi[ny + 4][i] - phi[ny + 3][i];
        }
        for (int j = 0; j < ny_total; ++j) {
            phi[j][2] = 2.0 * phi[j][3] - phi[j][4];
            phi[j][1] = 2.0 * phi[j][2] - phi[j][3];
            phi[j][0] = 2.0 * phi[j][1] - phi[j][2];
            phi[j][nx + 3] = 2.0 * phi[j][nx + 2] - phi[j][nx + 1];
            phi[j][nx + 4] = 2.0 * phi[j][nx + 3] - phi[j][nx + 2];
            phi[j][nx + 5] = 2.0 * phi[j][nx + 4] - phi[j][nx + 3];
        }
    }
    data = phi;
}

