#include "PowerLawField.hpp"
#include <iostream>

namespace HHG::Laser {
    PowerLawField::PowerLawField(h_float photon_energy, h_float E_0, h_float model_ratio, h_float duration, h_float exponent)
        : Laser(photon_energy, E_0, model_ratio, 0, duration, true), _exponent{exponent}
    {
        this->compute_spline();
    }

    h_float PowerLawField::envelope(h_float t) const {
        throw std::runtime_error("Enevelope of PowerLawField should never be called!");
    }

    void PowerLawField::compute_spline()
    {
        // Its just a template for the spline
        constexpr int N_spline = 256;
        std::array<h_float, N_spline> __temp;
        const h_float dt = t_end / N_spline;
        const h_float norm = _exponent > 0 ? 1. / std::pow(t_end, _exponent) : 1. / std::pow(dt, _exponent);
        __temp[0] = 0.;
        for (int i = 1; i < N_spline; ++i) {
            __temp[i] = momentum_amplitude * std::pow(i * dt, _exponent) * norm;
        }
        this->laser_spline = Spline(__temp.data(), __temp.size(), 0, t_end / (__temp.size() - 1));
    }
}