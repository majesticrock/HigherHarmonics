#pragma once
#include "Laser.hpp"
#include "ExperimentParams.hpp"
#include <iostream>

namespace HHG::Laser {

    template<class BaseLaser>
    struct DoubleLaser : public Laser {
        enum class Active { A, B, Both };

        // Converts a time in picoseconds to units of the inverse laser frequency
        constexpr static h_float ps_to_uniteless(h_float ps) {
            return ps * exp_photon_energy / (1e12 * hbar);
        }
        // Converts a time in units of the inverse laser frequency to picoseconds
        constexpr static h_float uniteless_to_ps(h_float unitless) {
            return unitless * (1e12 * hbar) / exp_photon_energy;
        }
        constexpr static h_float second_laser_phase_shift = 0.0;

        const h_float unified_t_max; // 1.5 * the duration of one pulse
        const h_float second_laser_shift{}; ///< in units of hbar omega

        // photon energy in units of the experimental frequency
        // E_0 in units of the experimental E_max
        DoubleLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float _second_laser_shift, Active _active_laser = Active::Both)
            : Laser(exp_photon_energy * photon_energy, E_0, model_ratio, 0., 1.5 * BaseLaser::base_duration(n_cycles), true),
            unified_t_max{1.5 * BaseLaser::base_duration(n_cycles)},
            second_laser_shift{photon_energy * ps_to_uniteless(_second_laser_shift)},
            active_laser{_active_laser},
            laserA(exp_photon_energy * photon_energy, E_0 * 1.6, model_ratio, n_cycles),
            laserB(exp_photon_energy * photon_energy, E_0 * 0.7, model_ratio, n_cycles, second_laser_phase_shift)
        {
            this->compute_spline();
        };
        
        h_float envelope(h_float t) const final
        {
            throw std::runtime_error("Envelope of the DoubleLaser should never be called!");
        };
    private:
        Active active_laser;

        BaseLaser laserA;
        BaseLaser laserB;

        void compute_spline() override final
            {
            constexpr int N = 1000;
            const h_float dt = unified_t_max / (N - 1);

            auto add_laser = [this](const h_float A_A, const h_float A_B) -> h_float {
                if (this->active_laser == Active::Both)
                    return A_A + A_B;
                if (this->active_laser == Active::A)
                    return A_A;
                else
                    return A_B;
            };

            std::vector<h_float> __temp(N);
            for(int i = 0; i < N; ++i) {
                const h_float t = i * dt;
                const h_float __A = laserA.laser_function(t);
                const h_float __B = laserB.laser_function(t - second_laser_shift + second_laser_phase_shift);
                __temp[i] = add_laser(__A, __B);
            }

            std::cout << "Max shift = " << std::ranges::max(__temp, [](double l, double r){return std::abs(l) < std::abs(r);}) << std::endl;

            this->laser_spline = Spline(__temp.data(), N, t_begin, dt, h_float{}, h_float{});
        };
    };
}