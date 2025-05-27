#include "DiracSystem.hpp"
#include <cmath>
#include <cassert>
#include <numeric>

#include <omp.h>
#include <nlohmann/json.hpp>

#include <mrock/utility/Numerics/Integration/AdaptiveTrapezoidalRule.hpp>
#include <mrock/utility/Numerics/ErrorFunctors.hpp>
#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/progress_bar.hpp>

constexpr HHG::h_float abs_error = 1.0e-12;
constexpr HHG::h_float rel_error = 1.0e-8;

#include "DiracDetail/MagnusMatrix.hpp"

#include <boost/numeric/odeint.hpp>
typedef HHG::Systems::DiracSystem::c_vector state_type;
typedef HHG::Systems::DiracSystem::sigma_vector sigma_state_type;
using namespace boost::numeric::odeint;

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

typedef runge_kutta_fehlberg78<state_type> error_stepper_type;
typedef runge_kutta_fehlberg78<sigma_state_type> sigma_error_stepper_type;

#ifdef NO_MPI
#define PROGRESS_BAR_UPDATE ++(progresses[omp_get_thread_num()]); \
            if (omp_get_thread_num() == 0) { \
                mrock::utility::progress_bar( \
                    static_cast<float>(std::reduce(progresses.begin(), progresses.end())) / static_cast<float>(n_z) \
                ); \
            }
#else
#define PROGRESS_BAR_UPDATE
#endif

namespace HHG::Systems {
    DiracSystem::DiracSystem(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy)
        : beta { is_zero(temperature) ? std::numeric_limits<h_float>::infinity() : _photon_energy / (k_B * temperature) },
        E_F{ _E_F / _photon_energy }, 
        v_F{ _v_F },
        band_width{ _band_width },
        max_k { band_width }, // in units of hbar omega_L
        max_kappa_compare { band_width * band_width }  // in units of (hbar omega_L)^2
    {  }

    DiracSystem::DiracSystem(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _decay_time)
        : beta { is_zero(temperature) ? std::numeric_limits<h_float>::infinity() : _photon_energy / (k_B * temperature) },
        E_F{ _E_F / _photon_energy }, 
        v_F{ _v_F },
        band_width{ _band_width },
        max_k { band_width }, // in units of hbar omega_L
        max_kappa_compare { band_width * band_width },  // in units of (hbar omega_L)^2
        inverse_decay_time { (1e15 * hbar) / (_decay_time * _photon_energy) }
    {  }

    void DiracSystem::time_evolution(nd_vector& rhos, Laser::Laser const * const laser, 
        h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const
    {
        auto right_side = [this, &laser, &k_z, &kappa](const state_type& alpha_beta, state_type& dxdt, const h_float t) {
            dxdt = this->dynamical_matrix(k_z, kappa, laser->laser_function(t)) * alpha_beta;
        };
        state_type current_state = { fermi_function(E_F + dispersion(k_z, kappa), beta), fermi_function(E_F - dispersion(k_z, kappa), beta) };
        const h_float inital_norm = current_state.norm();
        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();

        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;
        
        rhos.conservativeResize(time_config.n_measurements + 1);
        rhos(0) = std::norm(current_state(0)) - std::norm(current_state(1));
        
        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            current_state.normalize();      // We do not have relaxation, thus |alpha|^2 + |beta|^2 is a conserved quantity.
            current_state *= inital_norm;   // We enforce this explicitly

            rhos(i) = std::norm(current_state(0)) - std::norm(current_state(1));

            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void DiracSystem::time_evolution_complex(std::vector<h_complex>& alphas, std::vector<h_complex>& betas, Laser::Laser const * const laser, 
        h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config) const
    {
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
        
        for (int i = 0; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);

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
        const h_float magnitude_k = norm(k_z, kappa);
        auto right_side = [this, &laser, &k_z, &kappa, &magnitude_k](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const h_float vector_potential = laser->laser_function(t);          
            const h_float m_x = vector_potential * 2.0 * kappa / magnitude_k;
            const h_float m_z = 2.0 * (magnitude_k - vector_potential * k_z / magnitude_k);

            dxdt[0] = m_z * state[1];
            dxdt[1] = m_x * state[2] - state[0] * m_z;
            dxdt[2] = -m_x * state[1];
        };

        const h_float alpha_0 = fermi_function(E_F + dispersion(k_z, kappa), beta);
        const h_float beta_0 = fermi_function(E_F - dispersion(k_z, kappa), beta);

        sigma_state_type current_state = { 2. * alpha_0 * beta_0, h_float{0}, alpha_0 * alpha_0 - beta_0 * beta_0 };
        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();

        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.conservativeResize(time_config.n_measurements + 1);
        rhos[0] = current_state(2);
        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            rhos[i] = current_state(2);
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void DiracSystem::time_evolution_magnus(nd_vector &rhos, Laser::Laser const *const laser, h_float k_z, h_float kappa, const TimeIntegrationConfig &time_config) const
    {
        const h_float magnitude_k = norm(k_z, kappa);
        //std::cout << k_z << "  " << kappa << "  " << dispersion(k_z, kappa) << std::endl;
        const h_float alpha_0 = fermi_function(E_F + dispersion(k_z, kappa), beta);
        const h_float beta_0 = fermi_function(E_F - dispersion(k_z, kappa), beta);

        sigma_state_type current_state = { 2. * alpha_0 * beta_0, h_float{0}, alpha_0 * alpha_0 - beta_0 * beta_0 };
        
        const h_float measure_every = time_config.measure_every();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.conservativeResize(time_config.n_measurements + 1);
        rhos[0] = current_state(2);

        // Factor 2 due to the prefactor of the M matrix (see onenote)
        Magnus magnus(2 * magnitude_k, 2 * kappa, 2 * k_z, measure_every);

        std::array<h_float, 4> expansion_coefficients;
        for (int i = 1; i <= time_config.n_measurements; ++i) {
            expansion_coefficients = laser->magnus_coefficients(measure_every, t_begin);
            for (auto& coeff : expansion_coefficients) {
                coeff /= magnitude_k;
            }

            current_state.applyOnTheLeft(magnus.Omega(expansion_coefficients));

            rhos[i] = current_state(2);
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void DiracSystem::time_evolution_decay(nd_vector &rhos, Laser::Laser const *const laser, h_float k_z, h_float kappa, const TimeIntegrationConfig &time_config) const
    {
        const h_float magnitude_k = norm(k_z, kappa);
        const h_float alpha_0 = fermi_function(E_F + dispersion(k_z, kappa), beta);
        const h_float beta_0 = fermi_function(E_F - dispersion(k_z, kappa), beta);

        sigma_state_type current_state = { 2. * alpha_0 * beta_0, h_float{0}, alpha_0 * alpha_0 - beta_0 * beta_0 };
        const sigma_state_type initial_state = current_state;

        auto right_side = [this, &laser, &k_z, &kappa, &magnitude_k, &initial_state](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const h_float vector_potential = laser->laser_function(t);          
            const h_float m_x = vector_potential * 2.0 * kappa / magnitude_k;
            const h_float m_z = 2.0 * (magnitude_k - vector_potential * k_z / magnitude_k);

            dxdt[0] = m_z * state[1]                  - (state[0] - initial_state[0]) * inverse_decay_time;
            dxdt[1] = m_x * state[2] - state[0] * m_z - (state[1] - initial_state[1]) * inverse_decay_time;
            dxdt[2] = -m_x * state[1]                 - (state[2] - initial_state[2]) * inverse_decay_time;
        };

        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();

        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.conservativeResize(time_config.n_measurements + 1);
        rhos[0] = current_state(2);
        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            rhos[i] = current_state(2);
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

    std::vector<h_float> DiracSystem::compute_current_density(Laser::Laser const *const laser, TimeIntegrationConfig const &time_config, 
        const int rank, const int n_ranks, const int n_z, const int n_kappa, const h_float kappa_threshold) const
    {
        nd_vector rhos_buffer = nd_vector::Zero(time_config.n_measurements + 1);
        std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});

        auto integration_weight = [](h_float k_z, h_float kappa) {
            return k_z * kappa / norm(k_z, kappa);
        };

        const auto delta_z = 2 * z_integration_upper_limit() / n_z;
        mrock::utility::Numerics::Integration::adapative_trapezoidal_rule<
            h_float, mrock::utility::Numerics::Integration::adapative_trapezoidal_rule_print_policy{false, false, false}
            > integrator;
#ifdef NO_MPI
        std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for firstprivate(rhos_buffer) reduction(vec_plus:current_density_time) schedule(dynamic)
        for (int z = 1; z < n_z; ++z)
#else
        int jobs_per_rank = (n_z - 1) / n_ranks;
        if (jobs_per_rank * n_ranks < n_z - 1) ++jobs_per_rank;
        const int this_rank_min_z = rank * jobs_per_rank + 1;
        const int this_rank_max_z = this_rank_min_z + jobs_per_rank > n_z ? n_z : this_rank_min_z + jobs_per_rank;
        for (int z = this_rank_min_z; z < this_rank_max_z; ++z)
#endif
        {
            PROGRESS_BAR_UPDATE;

            if (z == n_z/2) continue; // integration weight (z = 0) = 0

            const auto k_z = (z - n_z / 2) * delta_z;
            auto kappa_integrand = [&](h_float kappa) -> const nd_vector& {
                if (is_zero(kappa)) {
                    rhos_buffer.setZero();
                }
                else {
                    //time_evolution_sigma(rhos_buffer, laser, k_z, kappa, time_config);
                    time_evolution_magnus(rhos_buffer, laser, k_z, kappa, time_config);
                    //time_evolution(rhos_buffer, laser, k_z, kappa, time_config);
                    rhos_buffer *= integration_weight(k_z, kappa);
                }
                return rhos_buffer;
            };

            auto kappa_result = integrator.split_integrate<100>(kappa_integrand, h_float{}, kappa_integration_upper_limit(k_z), 
                n_kappa, kappa_threshold, mrock::utility::Numerics::vector_elementwise_error<nd_vector, h_float, false>(), nd_vector::Zero(time_config.n_measurements + 1));

            std::transform(current_density_time.begin(), current_density_time.end(), kappa_result.begin(), current_density_time.begin(), std::plus<>());
        }
        std::cout << std::endl;  
        for (int i = 0; i <= time_config.n_measurements; ++i) {
            current_density_time[i] *= delta_z;
        }
        return current_density_time;
    }

    std::array<std::vector<h_float>, n_debug_points> DiracSystem::compute_current_density_debug(Laser::Laser const *const laser, TimeIntegrationConfig const &time_config, 
        const int n_z, const int n_kappa, const h_float kappa_threshold) const
    {
        auto integration_weight = [](h_float k_z, h_float kappa) {
            return k_z * kappa / norm(k_z, kappa);
        };
        const auto delta_z = 2 * z_integration_upper_limit() / n_z;

        // Debug setup
        std::array<nd_vector, n_debug_points> time_evolutions{};
        time_evolutions.fill(nd_vector::Zero(time_config.n_measurements + 1));

        const int picked_z = n_z / 2 + n_z / 10;
        const h_float picked_kappa_max = kappa_integration_upper_limit((picked_z - n_z / 2) * delta_z);
        std::array<h_float, n_debug_points> picked{};

        for (int i = 0; i < n_debug_points; ++i) {
            picked[i] = (i + 1) * picked_kappa_max / (n_debug_points + 1);
            time_evolution_magnus(time_evolutions[i], laser, (picked_z - n_z / 2) * delta_z, picked[i], time_config);
            time_evolutions[i] *= integration_weight((picked_z - n_z / 2) * delta_z, picked[i]);
        }
        // end debug setup

        std::array<std::vector<h_float>, n_debug_points> time_evolutions_std;
        for(int i = 0; i < n_debug_points; ++i) {
            time_evolutions_std[i].resize(time_config.n_measurements + 1);
            std::copy(time_evolutions[i].begin(), time_evolutions[i].end(), time_evolutions_std[i].begin());
        }
        return time_evolutions_std;
    }

    std::vector<h_float> DiracSystem::compute_current_density_decay(Laser::Laser const *const laser, TimeIntegrationConfig const &time_config,
         const int rank, const int n_ranks, const int n_z, const int n_kappa, const h_float kappa_threshold) const
    {
        nd_vector rhos_buffer = nd_vector::Zero(time_config.n_measurements + 1);
        std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});

        auto integration_weight = [](h_float k_z, h_float kappa) {
            return k_z * kappa / norm(k_z, kappa);
        };

        const auto delta_z = 2 * z_integration_upper_limit() / n_z;
        mrock::utility::Numerics::Integration::adapative_trapezoidal_rule<
            h_float, mrock::utility::Numerics::Integration::adapative_trapezoidal_rule_print_policy{false, false, false}
            > integrator;
#ifdef NO_MPI
        std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for firstprivate(rhos_buffer) reduction(vec_plus:current_density_time) schedule(dynamic)
        for (int z = 1; z < n_z; ++z)
#else
        int jobs_per_rank = (n_z - 1) / n_ranks;
        if (jobs_per_rank * n_ranks < n_z - 1) ++jobs_per_rank;
        const int this_rank_min_z = rank * jobs_per_rank + 1;
        const int this_rank_max_z = this_rank_min_z + jobs_per_rank > n_z ? n_z : this_rank_min_z + jobs_per_rank;
        for (int z = this_rank_min_z; z < this_rank_max_z; ++z)
#endif
        {
            PROGRESS_BAR_UPDATE;

            if (z == n_z/2) continue; // integration weight (z = 0) = 0

            const auto k_z = (z - n_z / 2) * delta_z;
            auto kappa_integrand = [&](h_float kappa) -> const nd_vector& {
                if (is_zero(kappa)) {
                    rhos_buffer.setZero();
                }
                else {
                    //time_evolution_sigma(rhos_buffer, laser, k_z, kappa, time_config);
                    time_evolution_decay(rhos_buffer, laser, k_z, kappa, time_config);
                    //time_evolution(rhos_buffer, laser, k_z, kappa, time_config);
                    rhos_buffer *= integration_weight(k_z, kappa);
                }
                return rhos_buffer;
            };

            auto kappa_result = integrator.split_integrate<100>(kappa_integrand, h_float{}, kappa_integration_upper_limit(k_z), 
                n_kappa, kappa_threshold, mrock::utility::Numerics::vector_elementwise_error<nd_vector, h_float, false>(), nd_vector::Zero(time_config.n_measurements + 1));

            std::transform(current_density_time.begin(), current_density_time.end(), kappa_result.begin(), current_density_time.begin(), std::plus<>());
        }
        std::cout << std::endl;  
        for (int i = 0; i <= time_config.n_measurements; ++i) {
            current_density_time[i] *= delta_z;
        }
        return current_density_time;
    }

    std::array<std::vector<h_float>, n_debug_points> DiracSystem::compute_current_density_decay_debug(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config,
        const int n_z, const int n_kappa/*  = 20 */, const h_float kappa_threshold/*  = 1e-3 */) const 
    {
        auto integration_weight = [](h_float k_z, h_float kappa) {
            return k_z * kappa / norm(k_z, kappa);
        };
        const auto delta_z = 2 * z_integration_upper_limit() / n_z;

        // Debug setup
        std::array<nd_vector, n_debug_points> time_evolutions{};
        time_evolutions.fill(nd_vector::Zero(time_config.n_measurements + 1));

        const int picked_z = n_z / 2 + n_z / 10;
        const h_float picked_kappa_max = kappa_integration_upper_limit((picked_z - n_z / 2) * delta_z);
        std::array<h_float, n_debug_points> picked{};

        for (int i = 0; i < n_debug_points; ++i) {
            picked[i] = (i + 1) * picked_kappa_max / (n_debug_points + 1);
            time_evolution_decay(time_evolutions[i], laser, (picked_z - n_z / 2) * delta_z, picked[i], time_config);
            time_evolutions[i] *= integration_weight((picked_z - n_z / 2) * delta_z, picked[i]);
        }
        // end debug setup

        std::array<std::vector<h_float>, n_debug_points> time_evolutions_std;
        for(int i = 0; i < n_debug_points; ++i) {
            time_evolutions_std[i].resize(time_config.n_measurements + 1);
            std::copy(time_evolutions[i].begin(), time_evolutions[i].end(), time_evolutions_std[i].begin());
        }
        return time_evolutions_std;
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
        const h_float factor = vector_potential / magnitude_k;
        const c_matrix A{ {-factor * k_z + magnitude_k, -factor * kappa},
                          {-factor * kappa, factor * k_z - magnitude_k} };
        return imaginary_unit * A;
    }

    DiracSystem::r_matrix DiracSystem::real_dynamical_matrix(h_float k_z, h_float kappa, h_float vector_potential) const
    {
        const h_float magnitude_k = norm(k_z, kappa);
        const h_float factor = vector_potential / magnitude_k;
        return r_matrix{ {-factor * k_z + magnitude_k, -factor * kappa},
                         {-factor * kappa,              factor * k_z - magnitude_k} };
    }
}