#pragma once

#include "CosineLaser.hpp"

namespace HHG::Laser {
    struct DoubleCosine : public Laser {
        enum class Active { A, B, Both };

        constexpr static h_float exp_photon_energy{ 5.889401182228545 }; ///< in meV [obtained by FFT of the measured electric field]
        // Converts a time in picoseconds to units of the inverse laser frequency
        constexpr static h_float ps_to_uniteless(h_float ps) {
            return ps * exp_photon_energy / (1e12 * hbar);
        }
        // Converts a time in units of the inverse laser frequency to picoseconds
        constexpr static h_float uniteless_to_ps(h_float unitless) {
            return unitless * (1e12 * hbar) / exp_photon_energy;
        }

        const h_float unified_t_max; // 1.5 * the duration of one pulse
        const h_float second_laser_shift{}; ///< in units of hbar omega

        // photon energy in units of the experimental frequency
        // E_0 in units of the experimental E_max
        DoubleCosine(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float _second_laser_shift, Active _active_laser = Active::Both);
        
        h_float envelope(h_float t) const final;
    private:
        Active active_laser;

        CosineLaser laserA;
        CosineLaser laserB;

        void compute_spline() override final;
    };
}