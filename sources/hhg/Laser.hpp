#pragma once

#include "GlobalDefinitions"
#include <cmath>

namespace hhg {
    struct Laser {
    public:
        const h_float frequency{};
        const h_float momentum_amplitude{}; // e A / (hbar c)

        Laser(h_float p_frequency, h_float p_momentum_amplitude);

        virtual h_float envelope(h_float t) const = 0;

        inline h_float laser_function(h_float t) const {
            return momentum_amplitude * envelope(t) * std::cos(frequency * t);
        }
    };
}