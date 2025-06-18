#pragma once

#include "Laser.hpp"

namespace HHG::Laser {
    struct QuenchedField : public Laser {
        QuenchedField(h_float photon_energy, h_float E_0, h_float model_ratio, h_float duration);

        h_float envelope(h_float t) const final;

    protected:
        void compute_spline() override final;
    };
}