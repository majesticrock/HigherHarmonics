#include "DiracSystem.hpp"
#include <cmath>
#include <cassert>
#include <map>
#include <numeric>

#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <nlohmann/json.hpp>

#include <mrock/utility/Numerics/Integration/AdaptiveTrapezoidalRule.hpp>
#include <mrock/utility/Numerics/ErrorFunctors.hpp>
#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/progress_bar.hpp>

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

typedef HHG::DiracSystem::c_vector state_type;
typedef HHG::DiracSystem::sigma_vector sigma_state_type;
using namespace boost::numeric::odeint;

#define adaptive_stepper
#ifdef adaptive_stepper
typedef runge_kutta_fehlberg78<state_type> error_stepper_type;
typedef runge_kutta_fehlberg78<sigma_state_type> sigma_error_stepper_type;
#else
typedef runge_kutta4<state_type> stepper_type;
typedef runge_kutta4<sigma_state_type> sigma_stepper_type;
#endif

constexpr HHG::h_float abs_error = 1.0e-12;
constexpr HHG::h_float rel_error = 1.0e-8;

#define SIGMA_Q kappa / magnitude_k
#define SIGMA_R ((k_z + magnitude_k) * (k_z + magnitude_k) - kappa * kappa) / (2 * magnitude_k * (k_z + magnitude_k))

#define GENERATE_DEBUG_DATA

namespace HHG {
    DiracSystem::DiracSystem(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy)
        : beta { is_zero(temperature) ? std::numeric_limits<h_float>::infinity() : 1. / (k_B * temperature * _photon_energy) },
        E_F{ _E_F / _photon_energy }, 
        v_F{ _v_F * ((1e12 * hbar) / _photon_energy) }, // 1e12 for conversion to pm; T_L = hbar / _photon_energy
        band_width{ _band_width },
        max_k { band_width }, // in units of omega_L / v_F
        max_kappa_compare { band_width * band_width }  // in units of (omega_L / v_F)^2
    {  }

    void DiracSystem::time_evolution(std::vector<h_float>& alphas, std::vector<h_float>& betas, Laser::Laser const * const laser, 
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

    void DiracSystem::time_evolution_complex(std::vector<h_complex>& alphas, std::vector<h_complex>& betas, Laser::Laser const * const laser, 
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

    void DiracSystem::time_evolution_sigma(nd_vector& rhos, Laser::Laser const * const laser, 
        h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const
    {
#ifndef adaptive_stepper
        sigma_stepper_type stepper;
#endif
        auto right_side = [this, &laser, &k_z, &kappa](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const h_float vector_potential = laser->laser_function(t);
            const h_float magnitude_k = norm(k_z, kappa);
            const h_float m_x = 2 * v_F * vector_potential * SIGMA_Q;
            const h_float m_z = 2 * (magnitude_k - v_F * vector_potential * SIGMA_R);

            dxdt[0] = m_z * state[1];
            dxdt[1] = m_x * state[2] - state[0] * m_z;
            dxdt[2] = -m_x * state[1];
        };

        const h_float alpha_0 = fermi_function(E_F + dispersion(k_z, kappa), beta);
        const h_float beta_0 = fermi_function(E_F - dispersion(k_z, kappa), beta);

        const h_float rho_x = alpha_0 * beta_0;
        const h_float rho_z = alpha_0 * alpha_0 - beta_0 * beta_0;

        sigma_state_type current_state = { h_float{0}, h_float{0}, h_float{1} };
        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();

        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.conservativeResize(time_config.n_measurements + 1);
        rhos[0] = rho_z;
        for (int i = 1; i <= time_config.n_measurements; ++i) {
#ifdef adaptive_stepper
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
#else
            integrate_const(stepper, right_side, current_state, t_begin, t_end, dt);
#endif
            rhos[i] = current_state[0] * rho_x + current_state[2] * rho_z; // factor of 2 does not matter
            t_begin = t_end;
            t_end += measure_every;
        }
    }


    h_float DiracSystem::dispersion(h_float k_z, h_float kappa) const
    {
        return norm(kappa, k_z); // v_F is already contained within the k values
    }

    h_float DiracSystem::kappa_integration_upper_limit(h_float k_z) const
    {
        const h_float k_z_squared = k_z * k_z;
        assert(max_kappa_compare >= k_z_squared);
        return sqrt(max_kappa_compare - k_z_squared);
    }

    h_float DiracSystem::convert_to_z_integration(h_float abscissa) const noexcept
    {
        return max_k * abscissa;
    }
    
    h_float DiracSystem::convert_to_kappa_integration(h_float abscissa, h_float k_z) const
    {
        return 0.5 * kappa_integration_upper_limit(k_z) * (abscissa + h_float{1});
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

    std::vector<h_float> Dirac::compute_current_density(DiracSystem const& system, Laser::Laser const * const laser, 
        TimeIntegrationConfig const& time_config, const int n_z, const int n_kappa /* = 500 */, const h_float kappa_threshold /* = 1e-3 */,
        std::string const& debug_dir /* = "" */)
    {
        nd_vector rhos_buffer = nd_vector::Zero(time_config.n_measurements + 1);
        std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});

        auto integration_weight = [](h_float k_z, h_float kappa) {
            return k_z * kappa / norm(k_z, kappa);
        };

#ifdef GENERATE_DEBUG_DATA
        assert(!debug_dir.empty());
        constexpr int num_picked_times = 10;
        const int pick_time = time_config.n_measurements / 2;
        const int pick_z = n_z / 5;

        const int measurements_per_cycle = (int) (2 * pi * time_config.n_measurements / (time_config.t_end - time_config.t_begin));

        std::array<nd_vector, num_picked_times> pick_current_density_z;
        pick_current_density_z.fill(nd_vector::Zero(n_z + 1));
        std::array<std::map<h_float, h_float>, num_picked_times> pick_current_density_kappa;
        std::array<std::map<h_float, h_float>, num_picked_times> pick_current_density_kappa_minus;
#endif

        const auto delta_z = 2 * system.z_integration_upper_limit() / n_z;
        mrock::utility::Numerics::Integration::adapative_trapezoidal_rule<h_float, mrock::utility::Numerics::Integration::adapative_trapezoidal_rule_print_policy{false, false}> integrator;

        std::vector<int> progresses(omp_get_max_threads(), int{});

#pragma omp parallel for firstprivate(rhos_buffer) reduction(vec_plus:current_density_time) schedule(dynamic)
        for (int z = 1; z < n_z; ++z) { // f(|z| = z_max) = 0
            ++(progresses[omp_get_thread_num()]);
            if (omp_get_thread_num() == 0) {
                mrock::utility::progress_bar( 
                    static_cast<float>(std::reduce(progresses.begin(), progresses.end())) / static_cast<float>(n_z)
                );
            }

            //std::cout << "z=" << z << std::endl;
            const auto k_z = (z - n_z / 2) * delta_z;
            auto kappa_integrand = [&](h_float kappa) -> const nd_vector& {
                if (is_zero(kappa)) {
                    rhos_buffer.setZero();
                }
                else {
                    system.time_evolution_sigma(rhos_buffer, laser, k_z, kappa, time_config);
                    rhos_buffer *= integration_weight(k_z, kappa);
                } 
#ifdef GENERATE_DEBUG_DATA
                if (z == n_z - pick_z) {
                    for (int t = 0; t < num_picked_times; ++t) {
                        pick_current_density_kappa[t][kappa] = rhos_buffer[pick_time + t * measurements_per_cycle / (2 * num_picked_times)];
                    }  
                }
                if (z == pick_z) {
                    for (int t = 0; t < num_picked_times; ++t) {
                        pick_current_density_kappa_minus[t][kappa] = rhos_buffer[pick_time + t * measurements_per_cycle / (2 * num_picked_times)];
                    }
                }
#endif
                return rhos_buffer;
            };

            auto kappa_result = integrator.integrate(kappa_integrand, h_float{}, system.kappa_integration_upper_limit(k_z), 
                n_kappa, kappa_threshold, mrock::utility::Numerics::vector_elementwise_error<nd_vector, h_float, false>(), nd_vector::Zero(time_config.n_measurements + 1));

            std::transform(current_density_time.begin(), current_density_time.end(), kappa_result.begin(), current_density_time.begin(), std::plus<>());

#ifdef GENERATE_DEBUG_DATA
            for (int t = 0; t < num_picked_times; ++t) {
                pick_current_density_z[t][z] = kappa_result[pick_time + t * measurements_per_cycle / (2 * num_picked_times)];
            }
#endif
        }
        std::cout << std::endl;

#ifdef GENERATE_DEBUG_DATA
        std::vector<h_float> k_zs(n_z + 1);
        for (int i = 0; i <= n_z; ++i) {
            k_zs[i] = (i - n_z / 2) * delta_z;
        }
        nlohmann::json debug_json {
            { "time", 				                mrock::utility::time_stamp() },
            { "k_zs",                               k_zs },
            { "pick_current_density_z",             pick_current_density_z },
            { "pick_current_density_kappa",         pick_current_density_kappa },
            { "pick_current_density_kappa_minus",   pick_current_density_kappa_minus },
        };
        mrock::utility::saveString(debug_json.dump(4), debug_dir + "debug_data.json.gz");
#endif
        
        for (int i = 0; i <= time_config.n_measurements; ++i) {
            current_density_time[i] *= delta_z;
        }
        return current_density_time;
    }
}