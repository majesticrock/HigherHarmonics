#pragma once

#include <boost/numeric/odeint.hpp>
typedef HHG::DiracSystem::c_vector state_type;
typedef HHG::DiracSystem::sigma_vector sigma_state_type;
using namespace boost::numeric::odeint;

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#define adaptive_stepper
#ifdef adaptive_stepper
typedef runge_kutta_fehlberg78<state_type> error_stepper_type;
typedef runge_kutta_fehlberg78<sigma_state_type> sigma_error_stepper_type;
#else
typedef runge_kutta4<state_type> stepper_type;
typedef runge_kutta4<sigma_state_type> sigma_stepper_type;
#endif

#define SIGMA_Q kappa / magnitude_k
#define SIGMA_R ((k_z + magnitude_k) * (k_z + magnitude_k) - kappa * kappa) / (2 * magnitude_k * (k_z + magnitude_k))

#ifndef NO_MPI
#define DONT_GENERATE_DEBUG_DATA
#endif

#ifndef DONT_GENERATE_DEBUG_DATA
#define _CREATE_DEBUG_CONTAINERS constexpr int num_picked_times = 10; \
    const int pick_time = time_config.n_measurements / 2; \
    const int pick_z = n_z / 5; \
    const int measurements_per_cycle = (int) (2 * pi * time_config.n_measurements / (time_config.t_end - time_config.t_begin)); \
    std::array<nd_vector, num_picked_times> pick_current_density_z; \
    pick_current_density_z.fill(nd_vector::Zero(n_z + 1)); \
    std::array<std::map<h_float, h_float>, num_picked_times> pick_current_density_kappa; \
    std::array<std::map<h_float, h_float>, num_picked_times> pick_current_density_kappa_minus;

#define _SAVE_DEBUG_KAPPA if (z == n_z - pick_z) { \
        for (int t = 0; t < num_picked_times; ++t) { \
            pick_current_density_kappa[t][kappa] = rhos_buffer[pick_time + t * measurements_per_cycle / (2 * num_picked_times)]; \
        } \
    } \
    if (z == pick_z) { \
        for (int t = 0; t < num_picked_times; ++t) { \
            pick_current_density_kappa_minus[t][kappa] = rhos_buffer[pick_time + t * measurements_per_cycle / (2 * num_picked_times)]; \
        } \
    }

#define _SAVE_DEBUG_K_Z for (int t = 0; t < num_picked_times; ++t) { \
        pick_current_density_z[t][z] = kappa_result[pick_time + t * measurements_per_cycle / (2 * num_picked_times)]; \
    }

#define _SAVE_DEBUG_TO_JSON std::vector<h_float> k_zs(n_z + 1); \
    for (int i = 0; i <= n_z; ++i) { \
        k_zs[i] = (i - n_z / 2) * delta_z; \
    } \
    nlohmann::json debug_json { \
        { "time", 				                mrock::utility::time_stamp() }, \
        { "k_zs",                               k_zs }, \
        { "pick_current_density_z",             pick_current_density_z }, \
        { "pick_current_density_kappa",         pick_current_density_kappa }, \
        { "pick_current_density_kappa_minus",   pick_current_density_kappa_minus }, \
    }; \
    mrock::utility::saveString(debug_json.dump(4), debug_dir + "debug_data.json.gz");

#else

#define _CREATE_DEBUG_CONTAINERS
#define _SAVE_DEBUG_KAPPA
#define _SAVE_DEBUG_K_Z
#define _SAVE_DEBUG_TO_JSON

#endif