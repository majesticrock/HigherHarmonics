#pragma once

#include "Laser.hpp"

namespace HHG::Laser {
    struct ExperimentalLaser : public Laser {
        enum class Active { A, B, Both };

        // Experimental data in ps [average temporal spacing of measurements]
        constexpr static h_float exp_dt{ 0.03318960199004975 };
        constexpr static int N_extra = 16;
        constexpr static int N_experiment = 201;
        constexpr static h_float laser_end{ 6.67111 + 2 * (N_extra - 1) * exp_dt }; ///< in ps [measured data ends at the first number, the second summand is the buffer for a smooth approach to 0]
        constexpr static h_float exp_photon_energy{ 5.889401182228545 }; ///< in meV [obtained by FFT of the measured electric field]
        constexpr static h_float unified_t_max{ 2. * laser_end }; ///< in ps; I just assume that I'll never use t0 > 0.5 * laser_end

        const h_float second_laser_shift{}; ///< in units of hbar omega

        // photon_energy and E_0 should be given in units of the experimental input
        // That is, they merely rescale the experimental data
        ExperimentalLaser(h_float _photon_energy, h_float _E_0, h_float model_ratio, h_float _second_laser_shift, Active _active_laser = Active::Both);

        h_float envelope(h_float t) const final;
    protected:
        void compute_spline() override final;
        
        Active active_laser;
    };
}