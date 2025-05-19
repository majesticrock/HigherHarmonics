#include "CosineLaser.hpp"

namespace HHG::Laser {
    CosineLaser::CosineLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles)
        : Laser(photon_energy, E_0, model_ratio, 0., 2 * HHG::pi * n_cycles), envelope_omega{1. / n_cycles}
    {}

     CosineLaser::CosineLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float begin_shift)
        : Laser(photon_energy, E_0, model_ratio, begin_shift, begin_shift + 2 * HHG::pi * n_cycles), envelope_omega{1. / n_cycles}
    {}

    h_float CosineLaser::envelope(h_float t) const {
        if (t <= t_begin || t >= t_end) return h_float{};
        return 0.5 * (1. - std::cos(envelope_omega * (t - t_begin)));
    }
}