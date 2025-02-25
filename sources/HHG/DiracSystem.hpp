#pragma once

#include <mrock/utility/InputFileReader.hpp>
#include <string>
#include <array>
#include "GlobalDefinitions.hpp"
#include "Laser.hpp"
#include "TimeIntegrationConfig.hpp"

namespace HHG {
    class DiracSystem {
    public:
        using c_vector = complex_vector<2>;
        using c_matrix = complex_matrix<2, 2>;
        using r_matrix = real_matrix<2, 2>;
        using diagonal_matrix = Eigen::DiagonalMatrix<h_float, 2>;

        using sigma_vector = std::array<h_float, 3>;

        DiracSystem() = delete;
        /**
         * @param _E_F Fermi energy in meV
         * @param _v_F Fermi velocity in m/s
         * @param _band_width in multiples of the photon energy
         * @param _photon_energy hbar omega_L in meV
         */
        DiracSystem(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy);

        void time_evolution(std::vector<h_float>& alphas, std::vector<h_float>& betas, Laser const * const laser, 
            h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const;

        void time_evolution_complex(std::vector<h_complex>& alphas, std::vector<h_complex>& betas, Laser const * const laser, 
            h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const;

        void time_evolution_sigma(std::vector<h_float>& rhos, Laser const * const laser, 
            h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const;

        std::string info() const;

        h_float dispersion(h_float k_z, h_float kappa) const;

        h_float convert_to_z_integration(h_float abscissa) const;
        h_float convert_to_kappa_integration(h_float abscissa, h_float k_z) const;
    private:
        const h_float beta{}; ///< in units of the 1 / photon energy
        const h_float E_F{}; ///< in units of the photon energy
        const h_float v_F{}; ///< in units of pm / T_L, where T_L = 1 / omega_L
        const h_float band_width{}; ///< in units of the photon energy
        const h_float max_k{}; ///< in units of omega_L / v_F
        const h_float max_kappa_compare{}; ///< in units of (omega_L / v_F)^2

        h_float max_kappa(h_float k_z) const;

        // the matrix V
        r_matrix basic_transformation(h_float k_z, h_float kappa) const;

        // M = i v_F * V * h * V^+
        // d/dt (alpha, beta)^T = M * (alpha, beta)^T
        c_matrix dynamical_matrix(h_float k_z, h_float kappa, h_float vector_potential) const;

        // M = i v_F * V * h * V^+
        // d/dt (alpha, beta)^T = M * (alpha, beta)^T
        // this function omits the factor of i - causing the matrix to be real
        r_matrix real_dynamical_matrix(h_float k_z, h_float kappa, h_float vector_potential) const;
    };
}