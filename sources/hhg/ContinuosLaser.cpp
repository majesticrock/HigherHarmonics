#include "ContinuousLaser.hpp"

namespace hhg {
    ContinuousLaser::ContinuousLaser(h_float frequency, h_float momentum_amplitude)
            : Laser(frequency, momentum_amplitude) {}

    h_float ContinuousLaser::envelope(c_float t) const {
        return h_float{1};
    }
}