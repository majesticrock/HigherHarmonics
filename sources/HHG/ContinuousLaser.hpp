#pragma once

#include "Laser.hpp"

namespace HHG {
    struct ContinuousLaser : public Laser {
        ContinuousLaser(h_float photon_energy, h_float E_0);

        h_float envelope(h_float t) const final;
    };
}