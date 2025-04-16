#pragma once

#include <string>
#include <array>
#include <Eigen/Dense>

#include "GlobalDefinitions.hpp"
#include "TimeIntegrationConfig.hpp"
#include "Laser/Laser.hpp"


namespace HHG {
    class PiFlux {
    public:
        struct momentum_type {
            h_float cos_x{};
            h_float cos_y{};
            h_float cos_z{};
            h_float z{};

            momentum_type() = default;
            momentum_type(h_float x, h_float y, h_float z) noexcept;

            void update(h_float x, h_float y, h_float z) noexcept;
            void update_x(h_float val) noexcept;
            void update_y(h_float val) noexcept;
            void update_z(h_float val) noexcept;

            inline bool is_dirac_point() const noexcept {
                return (is_zero(cos_x) && is_zero(cos_y) && is_zero(cos_z));
            }
        };

        PiFlux() = delete;
        PiFlux(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _decay_time);

        inline h_float laser_model_ratio() const {
            return lattice_constant;
        }

        void time_evolution_sigma(nd_vector& rhos, Laser::Laser const * const laser, 
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

        h_float alpha(const momentum_type &k, h_float t, Laser::Laser const * const laser) const;
        h_float xi(const momentum_type& k, h_float t, Laser::Laser const * const laser) const;

        std::array<std::array<h_float, 3>, 4> magnus_coefficients(const momentum_type& k, h_float delta_t, h_float t_0, Laser::Laser const * const laser) const;
    
        void __time_evolution__(nd_vector& rhos, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config) const;
    };
}