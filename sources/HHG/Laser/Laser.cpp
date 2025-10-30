#include "ContinuousLaser.hpp"
#include <iostream>

#include "gauss.hpp"

namespace HHG::Laser {
     /** A_0 = 1e11 * E_0 / photon_energy
     * For the Dirac system:
     * We have hbar in meV * s -> so giving photon energy (gamma = hbar omega) in meV is just fine
     * E_0 is given in MV/cm, requiring the factor of 10^8 to convert to V/m
     * v_F is given in m/s -> already fine
     * 
     * [v_F]*[E_0] = 10^11 (meV/s)
     * [v_F]*[E_0]/[gamma] = 10^11 / [gamma] (1/s) = 10^11 * [hbar] / [gamma^2] (gamma/hbar)
     * 
     * Then 1e8 hbar v_F E_0 / (photon_energy) is in eV (for the parameters of Wang around 30.09 eV)
     * 
     * => model_ratio = hbar v_F / photon_energy
     * 
     * 
     * For lattice systems:
     * d * e / hbar * (A/c) = d * e / hbar * (E / omega_L) = d * e * E / (gamma)
     */
    Laser::Laser(h_float photon_energy, h_float E_0, h_float model_ratio)
        : momentum_amplitude{model_ratio * 1e11 * E_0 / photon_energy}, photon_energy{photon_energy}
    { }

    Laser::Laser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float t_begin, h_float t_end)
        : momentum_amplitude{model_ratio * 1e11 * E_0 / photon_energy}, t_begin{t_begin}, t_end{t_end}, photon_energy{photon_energy}
    { }

    Laser::Laser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float t_begin, h_float t_end, bool _use_spline)
        : momentum_amplitude{model_ratio * 1e11 * E_0 / photon_energy}, t_begin{t_begin}, t_end{t_end}, photon_energy{photon_energy}, use_spline{_use_spline}
    { }

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

    void Laser::compute_spline()
    {
        constexpr int N = 48e3; // This way we can compute 8 cycles (8*2*pi ~ 64) to an accuracy of h^4= ~ 1e-12
        std::array<h_float, N> __temp;

        const h_float dt = (t_end - t_begin) / N;
        for (int i = 0; i < N; ++i) {
            const h_float t = t_begin + dt * i;
            __temp[i] = this->__laser_function__(t);
        }

        this->laser_spline = Spline(__temp.data(), N, t_begin, dt);
    }
}