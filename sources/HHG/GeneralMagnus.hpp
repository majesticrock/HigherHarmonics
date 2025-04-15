#pragma once
#include "GlobalDefinitions.hpp"
#include <array>

namespace HHG {
    struct GeneralMagnus {
        typedef Eigen::Matrix<h_float, 3, 3> u_matrix;

        u_matrix Omega(std::array<h_float, 3> const& coeffs1, std::array<h_float, 3> const& coeffs2, 
            std::array<h_float, 3> const& coeffs3, std::array<h_float, 3> const& coeffs4) const;
    };
}