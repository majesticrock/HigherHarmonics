#include "DoubleCosine.hpp"
#include <iostream>

namespace HHG::Laser {
    DoubleCosine::DoubleCosine(h_float photon_energy, h_float E_0, h_float model_ratio, h_float n_cycles,
            h_float _second_laser_shift, Active _active_laser/* = Active::Both*/)
        : Laser(exp_photon_energy * photon_energy, 
            E_0, 
            model_ratio, 
            0., 
            1.5 * 2. * HHG::pi * n_cycles,
        true),
        unified_t_max{1.5 * 2. * HHG::pi * n_cycles},
        second_laser_shift{photon_energy * ps_to_uniteless(_second_laser_shift)},
        active_laser{_active_laser},
        laserA(exp_photon_energy * photon_energy, E_0 * 1.6, model_ratio, n_cycles),
        laserB(exp_photon_energy * photon_energy, E_0 * 0.7, model_ratio, n_cycles)
    {
        this->compute_spline();
    }

    h_float DoubleCosine::envelope(h_float t) const
    {
        throw std::runtime_error("Envelope of the DoubleCosine should never be called!");
    }

    void DoubleCosine::compute_spline()
    {
        constexpr int N = 1000;
        const h_float dt = unified_t_max / (N - 1);

        auto add_laser = [this](const h_float A_A, const h_float A_B) -> h_float {
            if (this->active_laser == Active::Both)
                return A_A + A_B;
            if (this->active_laser == Active::A)
                return A_A;
            else
                return A_B;
        };

        std::vector<h_float> __temp(N);
        for(int i = 0; i < N; ++i) {
            const h_float t = i * dt;
            const h_float __A = laserA.laser_function(t);
            const h_float __B = laserB.laser_function(t - second_laser_shift);
            __temp[i] = add_laser(__A, __B);
        }

        std::cout << "Max shift = " << std::ranges::max(__temp, [](double l, double r){return std::abs(l) < std::abs(r);}) << std::endl;

        this->laser_spline = Spline(__temp.data(), N, t_begin, dt, h_float{}, h_float{});
    }
}
