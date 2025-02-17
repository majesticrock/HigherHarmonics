#include "DiracSystem.hpp"
#include <cmath>

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
    DiracSystem::DiracSystem(h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy)
        : E_F{ _E_F / _photon_energy }, 
        v_F{ _v_F * ((1e12 * hbar) / _photon_energy) }, // 1e12 for conversion to pm; T_L = hbar / _photon_energy
        band_width{ _band_width },
        max_k { band_width / v_F }, // in pm
        max_kappa_compare { band_width * band_width / (v_F * v_F) }  // in pm^2
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
        state_type current_state = { static_cast<h_float>(E_F > -dispersion(k_z, kappa)), static_cast<h_float>(E_F > dispersion(k_z, kappa)) };
        const h_float inital_norm = current_state.norm();
        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();

        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;
        
        alphas.resize(time_config.n_measurements + 1);
        betas.resize(time_config.n_measurements + 1);

        alphas[0] = std::norm(current_state(0));
        betas[0] = std::norm(current_state(1));
        
        for (int i = 0; i < time_config.n_measurements; ++i) {
#ifdef adaptive_stepper
            integrate_adaptive(make_controlled<error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
#else
            integrate_const(stepper, right_side, current_state, t_begin, t_end, dt);
#endif
            current_state.normalize();      // We do not have relaxation, thus |alpha|^2 + |beta|^2 is a conserved quantity.
            current_state *= inital_norm;   // We enforce this explicitly

            alphas[i + 1] = std::norm(current_state(0));
            betas[i + 1] = std::norm(current_state(1));

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
        if ( is_zero(kappa) ) return r_matrix::Zero();
        const h_float magnitude_k = norm(k_z, kappa);
        
        const h_float N_plus  = norm(kappa, k_z - magnitude_k);
        const h_float N_minus = norm(kappa, k_z + magnitude_k);

        r_matrix V;
        V << -kappa / N_minus,             -kappa / N_plus,
            (k_z + magnitude_k) / N_minus, (k_z - magnitude_k) / N_plus;
        return V;
    }

    DiracSystem::c_matrix DiracSystem::dynamical_matrix(h_float k_z, h_float kappa, h_float vector_potential) const
    {
        c_matrix h;
        h << k_z - vector_potential, kappa,
            kappa, -k_z + vector_potential;
        h *= imaginary_unit * v_F;

        const r_matrix V = basic_transformation(k_z, kappa);
        return V.adjoint() * h * V;
    }

    h_float DiracSystem::dispersion(h_float k_z, h_float kappa) const
    {
        return v_F * norm(kappa, k_z);
    }
}