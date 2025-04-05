#pragma once
#include "../GlobalDefinitions.hpp"

namespace HHG {
    struct Magnus {
        const h_float k, k2, k3, k4;
        const h_float kappa, kappa2, kappa4; // kappa3 is not used
        const h_float k_z, k_z2, k_z3, k_z4;

        Magnus(h_float _k, h_float _kappa, h_float _k_z, h_float delta_t);

        typedef Eigen::Matrix<h_complex, 3, 3> m_matrix;
        typedef Eigen::Matrix<h_float, 3, 3> u_matrix;

        u_matrix Omega(h_float alpha, h_float beta, h_float gamma, h_float delta) const;
    };
}