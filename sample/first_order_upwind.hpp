#pragma once
#include <vector>

void compute_first_order_upwind(
    std::vector<std::vector<double>>& G, 
    const std::vector<std::vector<double>>& u,
    const std::vector<std::vector<double>>& v,
    double dx,
    double dy,
    double dt,
    double S_L
);
