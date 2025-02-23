#include "DiracSystem.hpp"
#include <cmath>
#include <cassert>

#include <boost/numeric/odeint.hpp>


typedef HHG::DiracSystem::c_vector state_type;
using namespace boost::numeric::odeint;

#define adaptive_stepper
#ifdef adaptive_stepper
typedef runge_kutta_fehlberg78<state_type> error_stepper_type;
#else
typedef runge_kutta4<state_type> stepper_type;
#endif

constexpr HHG::h_float abs_error = 1.0e-12;
constexpr HHG::h_float rel_error = 1.0e-8;

namespace HHG {
    DiracSystem::DiracSystem(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy)
        : beta { is_zero(temperature) ? std::numeric_limits<h_float>::infinity() : 1. / (k_B * temperature * _photon_energy) },
        E_F{ _E_F / _photon_energy }, 
        v_F{ _v_F * ((1e12 * hbar) / _photon_energy) }, // 1e12 for conversion to pm; T_L = hbar / _photon_energy
        band_width{ _band_width },
        max_k { band_width }, // in units of omega_L / v_F
        max_kappa_compare { band_width * band_width }  // in units of (omega_L / v_F)^2
    {  }

    void DiracSystem::time_evolution(std::vector<h_float>& alphas, std::vector<h_float>& betas, Laser const * const laser, 
        h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const
    {
#ifndef adaptive_stepper
        stepper_type stepper;
#endif
        auto right_side = [this, &laser, &k_z, &kappa](const state_type& alpha_beta, state_type& dxdt, const h_float t) {
            dxdt = this->dynamical_matrix(k_z, kappa, laser->laser_function(t)) * alpha_beta;
        };
        state_type current_state = { fermi_function(E_F + dispersion(k_z, kappa), beta), fermi_function(E_F - dispersion(k_z, kappa), beta) };
        const h_float inital_norm = current_state.norm();
        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();

        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;
        
        alphas.resize(time_config.n_measurements);
        betas.resize(time_config.n_measurements);

        alphas[0] = std::norm(current_state(0));
        betas[0] = std::norm(current_state(1));
        
        for (int i = 1; i < time_config.n_measurements; ++i) {
#ifdef adaptive_stepper
            integrate_adaptive(make_controlled<error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
#else
            integrate_const(stepper, right_side, current_state, t_begin, t_end, dt);
#endif
            current_state.normalize();      // We do not have relaxation, thus |alpha|^2 + |beta|^2 is a conserved quantity.
            current_state *= inital_norm;   // We enforce this explicitly

            alphas[i] = std::norm(current_state(0));
            betas[i] = std::norm(current_state(1));

            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void DiracSystem::time_evolution_complex(std::vector<h_complex>& alphas, std::vector<h_complex>& betas, Laser const * const laser, 
        h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const
    {
#ifndef adaptive_stepper
        stepper_type stepper;
#endif

        auto right_side = [this, &laser, &k_z, &kappa](const state_type& alpha_beta, state_type& dxdt, const h_float t) {
            dxdt = this->dynamical_matrix(k_z, kappa, laser->laser_function(t)) * alpha_beta;
        };
        state_type current_state = { static_cast<h_float>(E_F > -dispersion(k_z, kappa)), static_cast<h_float>(E_F > dispersion(k_z, kappa)) };
        const h_float inital_norm = current_state.norm();
        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();

        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;
        
        alphas.resize(time_config.n_measurements + 1);
        betas.resize(time_config.n_measurements + 1);

        alphas[0] = current_state(0);
        betas[0]  = current_state(1);
        
        for (int i = 0; i < time_config.n_measurements; ++i) {
#ifdef adaptive_stepper
            integrate_adaptive(make_controlled<error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
#else
            integrate_const(stepper, right_side, current_state, t_begin, t_end, dt);
#endif
            current_state.normalize();      // We do not have relaxation, thus |alpha|^2 + |beta|^2 is a conserved quantity.
            current_state *= inital_norm;   // We enforce this explicitly

            alphas[i + 1] = current_state(0);
            betas[i + 1]  = current_state(1);

            t_begin = t_end;
            t_end += measure_every;
        }
    }

    h_float DiracSystem::dispersion(h_float k_z, h_float kappa) const
    {
        return norm(kappa, k_z); // v_F is already contained within the k values
    }

    h_float DiracSystem::convert_to_z_integration(h_float abscissa) const
    {
        return max_k * abscissa;
    }
    
    h_float DiracSystem::convert_to_kappa_integration(h_float abscissa, h_float k_z) const
    {
        const h_float k_z_squared = k_z * k_z;
        assert(max_kappa_compare >= k_z_squared);
        const h_float up = sqrt(max_kappa_compare - k_z_squared);
        return 0.5 * up * (abscissa + h_float{1});
    }

    std::string DiracSystem::info() const
    {
        return "DiracSystem\nE_F=" + std::to_string(E_F) + " * hbar omega_L"
            + "\nv_F=" + std::to_string(v_F) + " * pm / T_L"
            + "\nband_width=" + std::to_string(band_width)
            + "\nmax_k=" + std::to_string(max_k) + " pm"
            + "\nmax_kappa_compare=" + std::to_string(max_kappa_compare) + " pm^2";
    }

    h_float DiracSystem::max_kappa(h_float k_z) const
    {
        return (k_z > max_kappa_compare ? sqrt(max_kappa_compare - k_z) : h_float{});
    }

    DiracSystem::r_matrix DiracSystem::basic_transformation(h_float k_z, h_float kappa) const
    {
        const h_float magnitude_k = norm(k_z, kappa);
        if ( is_zero(magnitude_k) ) return r_matrix::Zero();
        const h_float N = 1. / sqrt(2 * magnitude_k * (k_z + magnitude_k));

        const r_matrix V{ {N * (k_z + magnitude_k), -N * kappa},
                          {N * kappa, N * (k_z + magnitude_k)} };
        return V;
    }

    DiracSystem::c_matrix DiracSystem::dynamical_matrix(h_float k_z, h_float kappa, h_float vector_potential) const
    {
        const h_float magnitude_k = norm(k_z, kappa);
        const h_float factor = v_F * vector_potential / magnitude_k;
        const c_matrix A{ {-factor * k_z + magnitude_k, -factor * kappa},
                          {-factor * kappa, factor * k_z - magnitude_k} };
        return imaginary_unit * A;
    }

    DiracSystem::r_matrix DiracSystem::real_dynamical_matrix(h_float k_z, h_float kappa, h_float vector_potential) const
    {
        const h_float magnitude_k = norm(k_z, kappa);
        const h_float factor = v_F * vector_potential / magnitude_k;
        return r_matrix{ {-factor * k_z + magnitude_k, -factor * kappa},
                         {-factor * kappa,              factor * k_z - magnitude_k} };
    }
}