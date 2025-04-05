#include "ContinuousLaser.hpp"
#include <boost/math/quadrature/gauss.hpp>

using integrator = boost::math::quadrature::gauss<HHG::h_float, 30>;

namespace HHG::Laser {
     /** 
     * converts e E_0 / (hbar omega_L) to 1 / pm
     * if E_0 is given in MV / cm and (hbar omega_L) in meV
     */
    constexpr h_float field_conversion = 1e-1; 

    Laser::Laser(h_float photon_energy, h_float E_0)
        : momentum_amplitude{field_conversion * E_0 / (photon_energy)} 
    {}

    Laser::Laser(h_float photon_energy, h_float E_0, h_float t_begin, h_float t_end)
        : momentum_amplitude{field_conversion * E_0 / (photon_energy)}, t_begin{t_begin}, t_end{t_end} 
    {}

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