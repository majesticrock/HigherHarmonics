#pragma once
#include "../GlobalDefinitions.hpp"
#include <array>

namespace HHG {
    struct Magnus {
        const h_float k, k2, k3, k4;
        const h_float kappa, kappa2, kappa4; // kappa3 is not used
        const h_float k_z, k_z2, k_z3, k_z4;

        Magnus(h_float _k, h_float _kappa, h_float _k_z, h_float delta_t);

        typedef Eigen::Matrix<h_complex, 3, 3> m_matrix;
        typedef Eigen::Matrix<h_float, 3, 3> u_matrix;

        u_matrix Omega(std::array<h_float, 4> const& expansion_coefficients) const;
    };
}