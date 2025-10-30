#pragma once

#include "Laser.hpp"

namespace HHG::Laser {
    struct GaussLaser : public Laser {
        GaussLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles);
        GaussLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float begin_shift);
        GaussLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float begin_shift, bool use_spline);

        h_float envelope(h_float t) const final;

        static constexpr h_float base_duration(h_float n_cycles) {
            constexpr h_float EXTEND{ 3.0 }; // extend the pulse duration by this factor to avoid edge effects
            return 2.0 * HHG::pi * EXTEND * n_cycles;
        }
    private:
        const h_float center;
        const h_float sigma;
    };
}