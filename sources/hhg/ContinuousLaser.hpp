#pragma once

#include "Laser.hpp"

namespace hhg {
    struct ContinuousLaser : public Laser {
        ContinuousLaser(h_float frequency, h_float momentum_amplitude);

        h_float envelope(c_float t) const final;
    };
}