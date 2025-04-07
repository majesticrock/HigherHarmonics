#include "ContinuousLaser.hpp"
#include <iostream>

#include "gauss.hpp"

namespace HHG::Laser {
     /** 
     * We have hbar in meV * s -> so giving photon energy (gamma = hbar omega) in meV is just fine
     * E_0 is given in MV/cm, requiring the factor of 10^8 to convert to V/m
     * v_F is given in m/s -> already fine
     * 
     * [v_F]*[E_0] = 10^11 (meV/s)
     * [v_F]*[E_0]/[gamma] = 10^11 / [gamma] (1/s) = 10^11 * [hbar] / [gamma^2] (gamma/hbar)
     * 
     * Then 1e8 hbar v_F E_0 / (photon_energy) is in eV (for the parameters of Wang around 30.09 eV)
     */

    Laser::Laser(h_float photon_energy, h_float E_0, h_float v_F)
        : momentum_amplitude{hbar * 1e11 * v_F * E_0 / (photon_energy * photon_energy)} 
    {}

    Laser::Laser(h_float photon_energy, h_float E_0, h_float v_F, h_float t_begin, h_float t_end)
        : momentum_amplitude{hbar * 1e11 * v_F * E_0 / (photon_energy * photon_energy)}, t_begin{t_begin}, t_end{t_end} 
    {
        //std::cout << momentum_amplitude * photon_energy << std::endl;
    }

    std::array<h_float, 4> Laser::magnus_coefficients(h_float delta_t, h_float t_0) const
    {
        std::array<h_float, 4> coeffs{};
        h_float laser_value;
        for (int i = 0; i < n_gauss; ++i) {
            laser_value = this->raw_laser_function(t_0 + delta_t * abscissa[i]);
            coeffs[0] += weights[i] * laser_value;
            coeffs[1] += weights[i] * legendre_2[i] * laser_value;
            coeffs[2] += weights[i] * legendre_3[i] * laser_value;
            coeffs[3] += weights[i] * legendre_4[i] * laser_value;
        }
        coeffs[0] *= this->momentum_amplitude;
        coeffs[1] *= 3. * this->momentum_amplitude;
        coeffs[2] *= 5. * this->momentum_amplitude;
        coeffs[3] *= 7. * this->momentum_amplitude;
        return coeffs;
    }
}