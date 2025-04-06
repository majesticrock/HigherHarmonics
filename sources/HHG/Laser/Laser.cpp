#include "ContinuousLaser.hpp"
#include <boost/math/quadrature/gauss.hpp>
#include <iostream>

using integrator = boost::math::quadrature::gauss<HHG::h_float, 30>;

namespace HHG::Laser {
     /** 
     * We have hbar in meV * s -> so giving photon energy in meV is just fine
     * E_0 is given in MV/cm, requiring the factor of 10^8 to convert to V/m
     * v_F is given in m/s -> already fine
     * 
     * Then 1e8 hbar v_F E_0 / (photon_energy) is in eV (for the parameters of Wang around 30.09 eV)
     */

    Laser::Laser(h_float photon_energy, h_float E_0, h_float v_F)
        : momentum_amplitude{hbar * 1e8 * v_F * E_0 / (photon_energy * photon_energy * 1e-3)} 
    {}

    Laser::Laser(h_float photon_energy, h_float E_0, h_float v_F, h_float t_begin, h_float t_end)
        : momentum_amplitude{hbar * 1e8 * v_F * E_0 / (photon_energy * photon_energy * 1e-3)}, t_begin{t_begin}, t_end{t_end} 
    {
        std::cout << momentum_amplitude * photon_energy << std::endl;
    }

    h_float Laser::magnus_1(h_float delta_t, h_float t_0) const
    {
        auto integrand = [&](h_float x) {
            return this->raw_laser_function(delta_t * x + t_0);
        };
        return integrator::integrate(integrand, 0.0, 1.0);
    }

    h_float Laser::magnus_2(h_float delta_t, h_float t_0) const
    {
        auto integrand = [&](h_float x) {
            return this->raw_laser_function(delta_t * x + t_0) * (2 * x - 1);
        };
        return integrator::integrate(integrand, 0.0, 1.0);
    }

    h_float Laser::magnus_3(h_float delta_t, h_float t_0) const
    {
        auto integrand = [&](h_float x) {
            return this->raw_laser_function(delta_t * x + t_0) * (6 * x * (x - 1) + 1);
        };
        return integrator::integrate(integrand, 0.0, 1.0);
    }

    h_float Laser::magnus_4(h_float delta_t, h_float t_0) const
    {
        auto integrand = [&](h_float x) {
            return this->raw_laser_function(delta_t * x + t_0) * (20 * x * x * x - 30 * x * x + 12 * x - 1);
        };
        return integrator::integrate(integrand, 0.0, 1.0);
    }
}