#pragma once

#include "Laser.hpp"

namespace HHG::Laser {
    struct PowerLawField : public Laser {
        PowerLawField(h_float photon_energy, h_float E_0, h_float model_ratio, h_float duration, h_float exponent);

        h_float envelope(h_float t) const final;

    protected:
        const h_float _exponent{};
        void compute_spline() override final;
    };
}