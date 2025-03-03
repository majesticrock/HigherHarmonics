#include "CosineLaser.hpp"

namespace HHG::Laser {
    CosineLaser::CosineLaser(h_float photon_energy, h_float E_0, h_float n_cycles)
        : Laser(photon_energy, E_0, 0., HHG::pi * n_cycles), envelope_omega{1. / n_cycles}
    {}

    h_float CosineLaser::envelope(h_float t) const {
        if (t <= t_begin || t >= t_end) return h_float{};
        return (1. - std::cos(envelope_omega * t));
    }
}