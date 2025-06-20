#pragma once
#include "../GlobalDefinitions.hpp"
#include "../TimeIntegrationConfig.hpp"
#include "../Laser/Laser.hpp"

#include <string>
#include <array>
#include <Eigen/Dense>
#include <cmath>
#include <complex>

#ifdef MROCK_CL1
#define __Z 240
#else
#define __Z 64
#endif

namespace HHG::Systems {
    class Honeycomb {
    private:
        static constexpr int N_k = __Z;
    public:
        constexpr static std::array< std::array<h_float, 2>, 3> nearest_neighbors = {
            std::array<h_float, 2>{ 1., 0. },
            std::array<h_float, 2>{ -0.5,  0.5 * sqrt_3 },
            std::array<h_float, 2>{ -0.5, -0.5 * sqrt_3 } 
        };

        struct momentum_type {
            h_float x{};
            h_float y{};
            h_complex gamma{};

            momentum_type() = default;
            momentum_type(h_float x, h_float y) noexcept;

            void set_gamma() noexcept;

            h_complex shifted_gamma(h_float x_shift, h_float y_shift) const noexcept;
            std::array<h_complex, 2> shifted_grad_gamma(h_float x_shift, h_float y_shift) const noexcept;

            void update(h_float x, h_float y) noexcept;
            void update_x(h_float val) noexcept;
            void update_y(h_float val) noexcept;

            inline bool is_dirac_point() const noexcept {
                return is_zero(gamma);
            }
            inline void invert() noexcept {
                this->x = -this->x;
                this->y = -this->y;
                this->gamma = std::conj(this->gamma);
            }
            momentum_type shift(h_float x_shift, h_float y_shift) const noexcept;
        };

        Honeycomb() = delete;
        Honeycomb(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _decay_time);
    
        inline h_float laser_model_ratio() const {
            return lattice_constant;
        }

        void time_evolution_sigma(nd_vector& rho_x, nd_vector& rho_y, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config) const;

        void time_evolution_decay(nd_vector& rho_x, nd_vector& rho_y, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config) const;

        std::vector<h_float> compute_current_density(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
            const int rank, const int n_ranks, const int n_z) const;
        
        std::array<std::vector<h_float>, n_debug_points> compute_current_density_debug(Laser::Laser const * const laser, 
            TimeIntegrationConfig const& time_config, const int n_z) const;

        std::string info() const;

        h_float dispersion(const momentum_type& k) const;

        std::string get_property_in_SI_units(const std::string& property, const h_float photon_energy) const;

    private:
        const h_float beta{}; ///< in units of the 1 / photon energy
        const h_float E_F{}; ///< in units of the photon energy
        const h_float hopping_element{}; ///< in units of the photon energy
        const h_float lattice_constant{}; ///< in 1/m
        const h_float inverse_decay_time{}; ///< in units of omega_L
        const h_float laser_theta{}; ///< Angle between the laser polarization and the x-axis, in radians
        const h_float cos_theta{ std::cos(laser_theta) };
        const h_float sin_theta{ std::sin(laser_theta) };

        void __time_evolution__(nd_vector& rho_x, nd_vector& rho_y, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config) const;

        h_float occupation_a(const momentum_type& k) const;
        h_float occupation_b(const momentum_type& k) const;

        Eigen::Vector<HHG::h_float, 3> ic_sigma(const momentum_type& k, h_float alpha_beta_diff, h_float alpha_beta_prod) const noexcept;
    };
}

#undef __Z