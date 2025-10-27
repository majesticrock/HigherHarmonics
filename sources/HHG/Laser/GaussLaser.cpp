#include "GaussLaser.hpp"

namespace HHG::Laser {
    GaussLaser::GaussLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles)
        : Laser(photon_energy, E_0, model_ratio, 0., base_duration(n_cycles)), 
          center{ 0.5 * base_duration(n_cycles) },
          sigma{ HHG::pi * n_cycles / std::sqrt(2.0 * std::log(1.0 / 0.05)) }
    {}

    GaussLaser::GaussLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float begin_shift)
        : Laser(photon_energy, E_0, model_ratio, begin_shift, begin_shift + base_duration(n_cycles)), 
          center{ 0.5 * begin_shift + HHG::pi * n_cycles },
          sigma{ HHG::pi * n_cycles / std::sqrt(2.0 * std::log(1.0 / 0.05)) }
    {}

    GaussLaser::GaussLaser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles, h_float begin_shift, bool use_spline)
        : Laser(photon_energy, E_0, model_ratio, begin_shift, begin_shift + base_duration(n_cycles), use_spline), 
          center{ 0.5 * begin_shift + HHG::pi * n_cycles },
          sigma{ HHG::pi * n_cycles / std::sqrt(2.0 * std::log(1.0 / 0.05)) }
    {
        this->compute_spline();
    }

    h_float GaussLaser::envelope(h_float t) const {
        return std::exp(-((t - center) * (t - center)) / (2.0 * sigma * sigma));
    }
}