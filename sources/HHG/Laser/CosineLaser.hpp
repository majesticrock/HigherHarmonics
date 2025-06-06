#pragma once

#include "Laser.hpp"

namespace HHG::Laser {
    struct CosineLaser : public Laser {
        CosineLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles);
        CosineLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float begin_shift);
        CosineLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float begin_shift, bool use_spline);

        h_float envelope(h_float t) const final;
    private:
        const h_float envelope_omega{};
    };
}