#pragma once

#include "Laser.hpp"

namespace HHG::Laser {
    struct ExperimentalLaser : public Laser {
        enum class Active { A, B, Both };

        constexpr static h_float laser_end{ 6.67111 }; ///< in ps [measured data ends here]
        constexpr static h_float exp_photon_energy{ 5.889401182228545 }; ///< in meV [obtained by FFT of the measured electric field]

        const h_float second_laser_shift{}; ///< in ps
        const h_float photon_energy{};

        // photon_energy and E_0 should be given in units of the experimental input
        // That is, they merely rescale the experimental data
        ExperimentalLaser(h_float _photon_energy, h_float _E_0, h_float model_ratio, h_float _second_laser_shift, Active _active_laser = Active::Both);

        h_float envelope(h_float t) const final;
    protected:
        void compute_spline() override final;
        
        Active active_laser;
    };
}