#pragma once

#include <string>
#include <array>
#include <Eigen/Dense>
#include <random>

#include "GlobalDefinitions.hpp"
#include "TimeIntegrationConfig.hpp"
#include "Laser/Laser.hpp"

#ifdef MROCK_CL1
#define __Z 256
#define __C 16
#else
#define __Z 64
#define __C 8
#endif

namespace HHG {
    class PiFlux {
    private:
        static constexpr int z_range = __Z;

        static constexpr int N_coarse = __C; // Must be divisible by 2.
        static constexpr int N_fine = 8 * N_coarse; // Must be divisible by 4. Otherwise the error integrator breaks

        static constexpr int n_xy_inner = 64;
    public:
        struct momentum_type {
            h_float cos_x{};
            h_float cos_y{};
            h_float cos_z{};
            h_float z{};

            momentum_type() = default;
            momentum_type(h_float x, h_float y, h_float z) noexcept;

            // Abused the symmetry; x -> -x and y -> -y yields the same result because everything depends only on cos x and cos y
            static momentum_type SymmetrizedRandom();
            template<typename Generator>
            static momentum_type SymmetrizedRandom(Generator& gen) {
                static std::uniform_real_distribution<h_float> dist_z(0.0, pi);
                static std::uniform_real_distribution<h_float> dist_xy(0.0, 0.5 * pi);
                return momentum_type(dist_xy(gen), dist_xy(gen), dist_z(gen));
            };

            void update(h_float x, h_float y, h_float z) noexcept;
            void update_x(h_float val) noexcept;
            void update_y(h_float val) noexcept;
            void update_z(h_float val) noexcept;

            inline bool is_dirac_point() const noexcept {
                return (is_zero(cos_x) && is_zero(cos_y) && is_zero(cos_z));
            }
            inline void invert() noexcept {
                this->z = -this->z;
            }
        };

        PiFlux() = delete;
        PiFlux(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _decay_time);

        inline h_float laser_model_ratio() const {
            return lattice_constant;
        }

        void time_evolution_sigma(nd_vector& rhos, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config) const;

        void time_evolution_decay(nd_vector& rhos, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config) const;

        void time_evolution_magnus(nd_vector& rhos, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config) const;

        std::array<std::vector<h_float>, n_debug_points> compute_current_density_debug(Laser::Laser const * const laser, 
            TimeIntegrationConfig const& time_config, const int n_z) const;

        std::vector<h_float> compute_current_density(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
            const int rank, const int n_ranks, const int n_z) const;
        
        std::string info() const;

        h_float dispersion(const momentum_type& k) const;

    private:
        const h_float beta{}; ///< in units of the 1 / photon energy
        const h_float E_F{}; ///< in units of the photon energy
        const h_float hopping_element{}; ///< in units of the photon energy
        const h_float lattice_constant{}; ///< in 1/m
        const h_float inverse_decay_time{}; ///< in units of omega_L

        h_float occupation_a(const momentum_type& k) const;
        h_float occupation_b(const momentum_type& k) const;

        h_float alpha(const momentum_type &k, h_float t, Laser::Laser const * const laser) const;
        h_float xi(const momentum_type& k, h_float t, Laser::Laser const * const laser) const;

        std::array<std::array<h_float, 3>, 4> magnus_coefficients(const momentum_type& k, h_float delta_t, h_float t_0, Laser::Laser const * const laser) const;
    
        void __time_evolution__(nd_vector& rhos, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config) const;

        nd_vector xy_integral(momentum_type& k, nd_vector& rhos_buffer, Laser::Laser const * const laser, TimeIntegrationConfig const& time_config) const;
        nd_vector improved_xy_integral(momentum_type& k, nd_vector& rhos_buffer, Laser::Laser const * const laser, TimeIntegrationConfig const& time_config) const;

        h_float ic_sigma_x(const momentum_type& k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept;
        h_float ic_sigma_y(const momentum_type& k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept;
        h_float ic_sigma_z(const momentum_type& k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept;

        std::vector<h_float> current_density_lattice_sum(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
            const int rank, const int n_ranks, const int n_z) const;
        std::vector<h_float> current_density_continuum_limit(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
            const int rank, const int n_ranks, const int n_z) const;

        std::vector<h_float> current_density_monte_carlo(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
            const int rank, const int n_ranks, const int n_z) const;
    };
}

#undef __Z
#undef __C