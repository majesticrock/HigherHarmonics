#pragma once

#include "GlobalDefinitions.hpp"
#include <cmath>

namespace HHG {
    /**
    * We measure the time in units of the period of the electric field
    * and therefore the energy in units of hbar omega_L
    */
    struct Laser {
    public:
        const h_float momentum_amplitude{}; // e E_0 / (hbar omega_L)

        /**
         * @param photon_energy \f$ \hbar \omega_L \f$ in meV
         * @param E_0 peak electric field strength in MV / cm
         */
        Laser(h_float photon_energy, h_float E_0);

        virtual h_float envelope(h_float t) const = 0;

        inline h_float laser_function(h_float t) const {
            return momentum_amplitude * envelope(t) * std::sin(t);
        }
    };
}