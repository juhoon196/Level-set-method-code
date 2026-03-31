#pragma once
#include <vector>
#include <algorithm>

class Field2D {
public:
    int nx, ny;  // 실제 내부 도메인 크기
    std::vector<std::vector<double>> data; // [ny+6][nx+6], ghost cell 3줄 포함

    Field2D(int nx_, int ny_);

    void fill(double value = 0.0);

    // ghost cell 3줄 periodic 경계 (좌우, 상하, 모서리)
    void apply_periodic();

    // anchored (하단 ghost는 초기값 고정), 나머지는 선형외삽
    void apply_anchored_bc(const Field2D& initial);

    // reinitialization (Osher-Sethian 방식 등, level-set용)
    void reinitialize(const Field2D& initial, double dx, double dy, int num_iters);

    // 다른 함수 필요시 여기에 추가
};

