#pragma once
#include "field.hpp"  // Field2D 선언 필요
#include <vector>

// --- 2D WENO5 미분 (X방향) ---
// 벡터 버전은 필요 없다면 제거해도 됨. (내부 구현에서만 사용하면 됨)

// --- G-equation RHS (upwind, SL*|gradG| 등 포함) ---
// 이건 그대로 써도 되고, Field2D로 완전히 통일해도 됨(코드 일관성 위해 추천)
void compute_geqn_rhs(
    const std::vector<std::vector<double>>& G,
    const std::vector<std::vector<double>>& u,
    const std::vector<std::vector<double>>& v,
    std::vector<std::vector<double>>& rhs,
    double dx, double dy, double S_L
);

// --- TVD RK3 3단계 전체 스텝 ---
// **Field2D 객체 버전**
void RK3_step(
    Field2D& G_field,
    const Field2D& u_field,
    const Field2D& v_field,
    double dx, double dy, double dt, double S_L,
    const Field2D& G_initial   // anchored boundary용!
);
