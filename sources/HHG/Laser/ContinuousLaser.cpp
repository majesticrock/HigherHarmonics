#include "ContinuousLaser.hpp"

namespace HHG::Laser {
    ContinuousLaser::ContinuousLaser(h_float photon_energy, h_float E_0)
        : Laser(photon_energy, E_0) {}

    h_float ContinuousLaser::envelope(h_float t) const {
        return h_float{1};
    }
}