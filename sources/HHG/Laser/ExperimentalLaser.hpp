#pragma once

#include "Laser.hpp"

namespace HHG::Laser {
    struct ExperimentalLaser : public Laser {
        enum class Active { A, B, Both };
        const h_float second_laser_shift{}; ///< in units of hbar omega
        const h_float lattice_constant{}; ///< in 1/m
        
        // photon_energy and E_0 should be given in units of the experimental input
        // That is, they merely rescale the experimental data
        ExperimentalLaser(h_float _photon_energy, h_float _E_0, h_float model_ratio, h_float _second_laser_shift, Active _active_laser = Active::Both);

        h_float envelope(h_float t) const final;
    protected:
        void compute_spline() override final;
        
        Active active_laser;
    };
}