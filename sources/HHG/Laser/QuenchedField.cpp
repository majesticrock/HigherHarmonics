#include "QuenchedField.hpp"
#include <iostream>

namespace HHG::Laser {
    QuenchedField::QuenchedField(h_float photon_energy, h_float E_0, h_float model_ratio, h_float duration)
        : Laser(photon_energy, E_0, model_ratio, 0, duration, true)
    {
        this->compute_spline();
    }

    h_float QuenchedField::envelope(h_float t) const {
        throw std::runtime_error("Enevelope of QuenchedField should never be called!");
    }

    void QuenchedField::compute_spline()
    {
        // Its just a template for the spline
        constexpr int N_spline = 256;
        std::array<h_float, N_spline> __temp;
        for (int i = 0; i < N_spline; ++i) {
            if (i < N_spline / 4 || i > 3 * N_spline / 4)
                __temp[i] = h_float{};
            else if (i == N_spline / 4 || i == 3 * N_spline / 4)
                __temp[i] = 0.5 * momentum_amplitude;
            else
                __temp[i] = momentum_amplitude;
        }
        this->laser_spline = Spline(__temp.data(), __temp.size(), 0, t_end / (__temp.size() - 1), h_float{}, h_float{});
    }
}