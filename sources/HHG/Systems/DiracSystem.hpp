#pragma once

#include <string>
#include <array>
#include <Eigen/Dense>

#include "../GlobalDefinitions.hpp"
#include "../TimeIntegrationConfig.hpp"
#include "../Laser/Laser.hpp"

namespace HHG::Systems {
    class DiracSystem {
    public:
        using c_vector = complex_vector<2>;
        using c_matrix = complex_matrix<2, 2>;
        using r_matrix = real_matrix<2, 2>;
        using diagonal_matrix = Eigen::DiagonalMatrix<h_float, 2>;

        using sigma_vector = Eigen::Vector<h_float, 3>;

        DiracSystem() = delete;
        /**
         * @param _E_F Fermi energy in meV
         * @param _v_F Fermi velocity in m/s
         * @param _band_width in multiples of the photon energy
         * @param _photon_energy hbar omega_L in meV
         */
        DiracSystem(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy);
        DiracSystem(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _decay_time);

        inline h_float laser_model_ratio(h_float photon_energy) const {
            return hbar * v_F / photon_energy;
        }

        void time_evolution(nd_vector& rhos, Laser::Laser const * const laser, 
            h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const;

        void time_evolution_complex(std::vector<h_complex>& alphas, std::vector<h_complex>& betas, Laser::Laser const * const laser, 
            h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const;

        void time_evolution_sigma(nd_vector& rhos, Laser::Laser const * const laser, 
            h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const;

        void time_evolution_magnus(nd_vector& rhos, Laser::Laser const * const laser, 
            h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const;

        void time_evolution_decay(nd_vector& rhos, Laser::Laser const * const laser, 
            h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const;

        std::string info() const;

        h_float dispersion(h_float k_z, h_float kappa) const;

        h_float kappa_integration_upper_limit(h_float k_z) const;
        inline h_float z_integration_upper_limit() const noexcept { return max_k; }
        h_float convert_to_z_integration(h_float abscissa) const noexcept;
        h_float convert_to_kappa_integration(h_float abscissa, h_float k_z) const;

        std::vector<h_float> compute_current_density(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
            const int rank, const int n_ranks, const int n_z, const int n_kappa = 20, const h_float kappa_threshold = 1e-3) const;

        std::array<std::vector<h_float>, n_debug_points> compute_current_density_debug(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config,
            const int n_z, const int n_kappa = 20, const h_float kappa_threshold = 1e-3) const;

        std::vector<h_float> compute_current_density_decay(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config,
            const int rank, const int n_ranks, const int n_z, const int n_kappa = 20, const h_float kappa_threshold = 1e-3) const;

        std::array<std::vector<h_float>, n_debug_points> compute_current_density_decay_debug(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config,
            const int n_z, const int n_kappa = 20, const h_float kappa_threshold = 1e-3) const;
    private:
        const h_float beta{}; ///< in units of the 1 / photon energy
        const h_float E_F{}; ///< in units of the photon energy
        const h_float v_F{}; ///< in units of pm / T_L, where T_L = 1 / omega_L
        const h_float band_width{}; ///< in units of the photon energy
        const h_float max_k{}; ///< in units of omega_L / v_F
        const h_float max_kappa_compare{}; ///< in units of (omega_L / v_F)^2
        const h_float inverse_decay_time{}; ///< in units of omega_L

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