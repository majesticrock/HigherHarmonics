#include "PiFlux.hpp"

#include "../GeneralMagnus.hpp"
#include "../Laser/gauss.hpp"
#include "../thread_gauss.hpp"

#include <boost/numeric/odeint.hpp>
#include <mrock/utility/progress_bar.hpp>

#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>
#include <omp.h>
#include <random>
#include <type_traits>
using namespace boost::numeric::odeint;

typedef runge_kutta_fehlberg78<HHG::Systems::PiFlux::sigma_state_type> sigma_error_stepper_type;

constexpr HHG::h_float abs_error = 1.0e-8;
constexpr HHG::h_float rel_error = 1.0e-8;

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : std::transform(                 \
        omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size(), decltype(omp_orig)::value_type{}))

#if defined(NO_MPI) && !defined(MROCK_CL1_CASCADE)
#define PROGRESS_BAR_UPDATE(z_max)                                                                           \
    ++(progresses[omp_get_thread_num()]);                                                                    \
    if (omp_get_thread_num() == 0) {                                                                         \
        mrock::utility::progress_bar(static_cast<float>(std::reduce(progresses.begin(), progresses.end())) / \
                                     static_cast<float>((z_max)));                                           \
    }
#else
#define PROGRESS_BAR_UPDATE(z_max)
#endif

// #define INTEGRATION_ERROR
#ifdef INTEGRATION_ERROR
#define INTEGRATOR_TYPEDEF(N)                  \
    using __gauss = gauss::container<2 * (N)>; \
    using __error = gauss::container<(N)>;
#define ERROR_INTEGRATOR_WEIGHT h_float(!(i & 1)) * transform_weight* __error::weights[i / 2]
#else
#define INTEGRATOR_TYPEDEF(N) using __gauss = gauss::container<2 * (N)>;
#define ERROR_INTEGRATOR_WEIGHT \
    h_float {}
#endif

// #define DEBUG_INTEGRATE
constexpr double RESCUE_TRAFO = 1e-4;

namespace HHG::Systems {
PiFlux::PiFlux(h_float temperature,
               h_float _E_F,
               h_float _v_F,
               h_float _band_width,
               h_float _photon_energy,
               h_float _diagonal_relaxation_time,
               h_float _offdiagonal_relaxation_time)
    : beta(is_zero(temperature) ? std::numeric_limits<h_float>::max() : _photon_energy / (k_B * temperature)),
      E_F(_E_F / _photon_energy),
      hopping_element(_band_width / sqrt_12),
      lattice_constant(sqrt_3 * hbar * _v_F / (_photon_energy * _band_width)),
      inverse_diagonal_relaxation_time((1e15 * hbar) / (_diagonal_relaxation_time * _photon_energy)),
      inverse_offdiagonal_relaxation_time((1e15 * hbar) / (_offdiagonal_relaxation_time * _photon_energy)) {
    // gauss::precompute<100>();
    // abort();
}

void PiFlux::time_evolution_sigma(
    nd_vector& rhos,
    Laser::Laser const* const laser,
    const momentum_type& k,
    const TimeIntegrationConfig& time_config) const
{
    const h_float prefactor = 4 * hopping_element;

    const h_float alpha_beta_diff = occupation_a(k) - occupation_b(k);

    sigma_state_type current_state = {
        ic_sigma_x(k, alpha_beta_diff),
        ic_sigma_y(k, alpha_beta_diff),
        ic_sigma_z(k, alpha_beta_diff)
    };

    auto right_side = [&k, &laser, &prefactor](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
        const sigma_state_type m = {
            k.cos_x,
            k.cos_y,
            std::cos(k.z - laser->laser_function(t))
        };

        dxdt = cross_product(m, state);

        for (auto& d : dxdt)
            d *= prefactor;
    };

    const h_float measure_every = time_config.measure_every();
    const h_float dt = time_config.dt();

    h_float t_begin = time_config.t_begin;
    h_float t_end = t_begin + measure_every;

    rhos.conservativeResize(time_config.n_measurements + 1);
    rhos[0] = current_state[2];

    for (int i = 1; i <= time_config.n_measurements; ++i) {
        integrate_adaptive(
            make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);

        rhos[i] = current_state[2];

        t_begin = t_end;
        t_end += measure_every;
    }
}

void PiFlux::time_evolution_diagonal_relaxation(
    nd_vector& rhos,
    Laser::Laser const* const laser,
    const momentum_type& k,
    const TimeIntegrationConfig& time_config) const
{
    const h_float prefactor = 4 * hopping_element;

    const h_float alpha_beta_diff =
        occupation_a(k) - occupation_b(k);

    sigma_state_type current_state = {
        ic_sigma_x(k, alpha_beta_diff),
        ic_sigma_y(k, alpha_beta_diff),
        ic_sigma_z(k, alpha_beta_diff)
    };

    sigma_state_type relax_to_equilibrium{};

    auto update_equilibrium_state = [&](const h_float laser_at_t) {
        auto shifted_k = k;
        shifted_k.update_z(k.z - laser_at_t);

        const h_float epsilon = dispersion(shifted_k);

        // At a Dirac point, the thermal density matrix has zero
        // expectation value for all three Pauli matrices.
        if (is_zero(epsilon)) {
            relax_to_equilibrium = {0., 0., 0.};
            return;
        }

        const h_float occupation_diff =
            occupation_a(shifted_k) - occupation_b(shifted_k);

        relax_to_equilibrium = {
            occupation_diff * shifted_k.cos_x / epsilon,
            occupation_diff * shifted_k.cos_y / epsilon,
            occupation_diff * shifted_k.cos_z / epsilon
        };
    };

    update_equilibrium_state(
        laser->laser_function(time_config.t_begin));

    auto right_side = [this, &k, &laser, &prefactor, &update_equilibrium_state, &relax_to_equilibrium](
            const sigma_state_type& state,
            sigma_state_type& dxdt,
            const h_float t)
    {
        const h_float laser_at_t = laser->laser_function(t);

        const sigma_state_type m = {
            k.cos_x,
            k.cos_y,
            std::cos(k.z - laser_at_t)
        };

        update_equilibrium_state(laser_at_t);

        dxdt = cross_product(m, state);

        for (std::size_t i = 0U; i < dxdt.size(); ++i) {
            // prefactor = 4*t
            dxdt[i] *= prefactor;

            // Relax sigma^x and sigma^y toward thermal equilibrium.
            if (i < 2U) {
                dxdt[i] -= inverse_offdiagonal_relaxation_time
                         * (state[i] - relax_to_equilibrium[i]);
            }
            // Relax sigma^z toward thermal equilibrium.
            else {
                dxdt[i] -= inverse_diagonal_relaxation_time
                         * (state[i] - relax_to_equilibrium[i]);
            }
        }
    };

    const h_float measure_every = time_config.measure_every();
    const h_float dt = time_config.dt();

    h_float t_begin = time_config.t_begin;
    h_float t_end = t_begin + measure_every;

    rhos.resize(time_config.n_measurements + 1);
    rhos[0] = current_state[2];

    for (int i = 1; i <= time_config.n_measurements; ++i) {
        integrate_adaptive(
            make_controlled<sigma_error_stepper_type>(
                abs_error, rel_error),
            right_side, current_state, t_begin, t_end, dt);

        rhos[i] = current_state[2];

        t_begin = t_end;
        t_end += measure_every;
    }
}

void PiFlux::evolve_occupation_numbers(
    std::vector<OccupationContainer::occupation_t>& occupations,
    Laser::Laser const* const laser,
    const momentum_type& k,
    const TimeIntegrationConfig& time_config,
    bool diagonal /* = true */) const
{
    const h_float prefactor = 4. * hopping_element;

    auto compute_occupations =
        [this, diagonal](
            const sigma_state_type& input,
            const momentum_type& momentum)
    {
        const auto sigma = diagonal
            ? diagonal_sigma(input, momentum)
            : input;

        const h_float sigma_norm =
            norm(sigma[0], sigma[1], sigma[2]);

        const h_float occupation_a =
            std::sqrt(std::max(
                h_float{},
                0.5 * (-sigma[2] + sigma_norm)));

        const h_float occupation_b =
            std::sqrt(std::max(
                h_float{},
                0.5 * (sigma[2] + sigma_norm)));

        return OccupationContainer::occupation_t{
            occupation_a, occupation_b
        };
    };

    const h_float alpha_beta_diff =
        occupation_a(k) - occupation_b(k);

    sigma_state_type current_state = {
        ic_sigma_x(k, alpha_beta_diff),
        ic_sigma_y(k, alpha_beta_diff),
        ic_sigma_z(k, alpha_beta_diff)
    };

    auto shifted_momentum = [&](const h_float laser_at_t) {
        auto shifted_k = k;
        shifted_k.update_z(k.z - laser_at_t);
        return shifted_k;
    };

    auto equilibrium_state = [&](const momentum_type& shifted_k) {
        sigma_state_type target{};

        const h_float epsilon = dispersion(shifted_k);

        // At a Dirac point, the thermal Pauli expectation values vanish.
        if (is_zero(epsilon))
            return target;

        const h_float occupation_diff =
            occupation_a(shifted_k) - occupation_b(shifted_k);

        target[0] = occupation_diff * shifted_k.cos_x / epsilon;
        target[1] = occupation_diff * shifted_k.cos_y / epsilon;
        target[2] = occupation_diff * shifted_k.cos_z / epsilon;

        return target;
    };

    auto right_side =
        [this, &k, &laser, &prefactor, &shifted_momentum,
         &equilibrium_state](
            const sigma_state_type& state,
            sigma_state_type& dxdt,
            const h_float t)
    {
        const h_float laser_at_t = laser->laser_function(t);
        const auto shifted_k = shifted_momentum(laser_at_t);
        const auto target = equilibrium_state(shifted_k);

        const sigma_state_type m = {
            k.cos_x,
            k.cos_y,
            shifted_k.cos_z
        };

        dxdt = cross_product(m, state);

        for (std::size_t i = 0U; i < dxdt.size(); ++i) {
            dxdt[i] *= prefactor;

            if (i < 2U) {
                // Relax sigma^x and sigma^y.
                dxdt[i] -= inverse_offdiagonal_relaxation_time
                         * (state[i] - target[i]);
            } else {
                // Relax sigma^z.
                dxdt[i] -= inverse_diagonal_relaxation_time
                         * (state[i] - target[i]);
            }
        }
    };

    const h_float measure_every = time_config.measure_every();
    const h_float dt = time_config.dt();

    h_float t_begin = time_config.t_begin;
    h_float t_end = t_begin + measure_every;

    occupations.resize(time_config.n_measurements + 1);

    auto measurement_momentum = shifted_momentum(
        laser->laser_function(time_config.t_begin));

    occupations[0] = compute_occupations(
        current_state, measurement_momentum);

    for (int i = 1; i <= time_config.n_measurements; ++i) {
        integrate_adaptive(
            make_controlled<sigma_error_stepper_type>(
                abs_error, rel_error),
            right_side, current_state, t_begin, t_end, dt);

        // Evaluate at the actual measurement time. Adaptive integrators
        // may evaluate the RHS at times other than t_end.
        measurement_momentum = shifted_momentum(
            laser->laser_function(t_end));

        occupations[i] = compute_occupations(
            current_state, measurement_momentum);

        t_begin = t_end;
        t_end += measure_every;
    }
}

std::array<std::vector<h_float>, n_debug_points> PiFlux::compute_current_density_debug(
    Laser::Laser const* const laser,
    TimeIntegrationConfig const& time_config,
    [[maybe_unused]] const int n_z) const {
    // Debug setup
    std::array<nd_vector, n_debug_points> time_evolutions{};
    time_evolutions.fill(nd_vector::Zero(time_config.n_measurements + 1));

    const h_float picked_z = 0.49 * pi;
    [[maybe_unused]] const h_float picked_x = 0.5 * pi;
    std::array<h_float, n_debug_points> picked{};

#ifdef DEBUG_INTEGRATE
    constexpr int n_gauss = 100;
    typedef gauss::container<2 * n_gauss> y_gauss;
    typedef gauss::container<n_gauss> error_gauss;
    std::array<nd_vector, n_debug_points> error_evolutions{};
    error_evolutions.fill(nd_vector::Zero(time_config.n_measurements + 1));
#endif

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_debug_points; ++i) {
        picked[i] = 0.5 * pi - (i * 0.02 * pi);  //(i + 1) * 0.5 * pi / (n_debug_points);
#ifdef DEBUG_INTEGRATE
        momentum_type k(picked[i], 0.0, picked_z);
        nd_vector rho_buffer = nd_vector::Zero(time_config.n_measurements + 1);
        for (int y = 0; y < n_gauss; ++y) {
            k.update_y(0.5 * pi * (0.5 + 0.5 * y_gauss::abscissa[y]));
            __time_evolution__(rho_buffer, laser, k, time_config);
            rho_buffer *= y_gauss::weights[y];
            time_evolutions[i] += rho_buffer;

            k.update_y(0.5 * pi * (0.5 - 0.5 * y_gauss::abscissa[y]));
            __time_evolution__(rho_buffer, laser, k, time_config);
            rho_buffer *= y_gauss::weights[y];
            time_evolutions[i] += rho_buffer;
        }
        for (int y = 0; y < n_gauss / 2; ++y) {
            k.update_y(0.5 * pi * (0.5 + 0.5 * error_gauss::abscissa[y]));
            __time_evolution__(rho_buffer, laser, k, time_config);
            rho_buffer *= error_gauss::weights[y];
            error_evolutions[i] += rho_buffer;

            k.update_y(0.5 * pi * (0.5 - 0.5 * error_gauss::abscissa[y]));
            __time_evolution__(rho_buffer, laser, k, time_config);
            rho_buffer *= error_gauss::weights[y];
            error_evolutions[i] += rho_buffer;
        }
        std::cout << "#" << i << "  k_y=" << picked[i] << ":    " << (error_evolutions[i] - time_evolutions[i]).norm()
                  << std::endl;
#else
        momentum_type k(picked_x, picked[i], picked_z);
        __time_evolution__(time_evolutions[i], laser, k, time_config);
#endif
    }
    // end debug setup

    std::array<std::vector<h_float>, n_debug_points> time_evolutions_std;
    for (int i = 0; i < n_debug_points; ++i) {
        time_evolutions_std[i].resize(time_config.n_measurements + 1);
        std::copy(time_evolutions[i].begin(), time_evolutions[i].end(), time_evolutions_std[i].begin());
    }
    return time_evolutions_std;
}

std::vector<h_float> PiFlux::compute_current_density(Laser::Laser const* const laser,
                                                     TimeIntegrationConfig const& time_config,
                                                     const int rank,
                                                     const int n_ranks,
                                                     const int n_z) const {
    // return current_density_lattice_sum(laser, time_config, rank, n_ranks, n_z);
    return current_density_continuum_limit(laser, time_config, rank, n_ranks, n_z);
    // return current_density_monte_carlo(laser, time_config, rank, n_ranks, n_z);
}

std::string PiFlux::info() const {
    return "PiFlux\nT=" + std::to_string(1.0 / beta) + "\n" + "E_F=" + std::to_string(E_F) + "\n" +
           "t=" + std::to_string(hopping_element) + "\n" + "d=" + std::to_string(lattice_constant) + "\n";
}

h_float PiFlux::dispersion(const momentum_type& k) const {
    return std::sqrt(k.cos_x * k.cos_x + k.cos_y * k.cos_y + k.cos_z * k.cos_z);
}

std::string PiFlux::get_property_in_SI_units(const std::string& property, const h_float photon_energy) const {
    if (property == "E_F") {
        return std::to_string(E_F * photon_energy) + " meV";
    } else if (property == "t") {
        return std::to_string(hopping_element * photon_energy) + " meV";
    } else if (property == "d") {
        return std::to_string(1e12 * lattice_constant) + " pm";
    } else if (property == "beta") {
        return std::to_string(beta / photon_energy) + " meV^-1";
    } else if (property == "T") {
        return std::to_string(photon_energy / (k_B * beta)) + " K";
    } else {
        throw std::invalid_argument("Property '" + property + "' is not recognized!");
    }
}

h_float PiFlux::occupation_a(const momentum_type& k) const {
    return fermi_function(-E_F + 2 * hopping_element * dispersion(k), beta);
}

h_float PiFlux::occupation_b(const momentum_type& k) const {
    return fermi_function(-E_F - 2 * hopping_element * dispersion(k), beta);
}

h_float PiFlux::alpha(const momentum_type& k, h_float t, Laser::Laser const* const laser) const {
    return k.cos_z - std::cos(k.z - laser->laser_function(t));
}

h_float PiFlux::xi(const momentum_type& k, h_float t, Laser::Laser const* const laser) const {
    return k.cos_x * k.cos_x + k.cos_y * k.cos_y + k.cos_z * std::cos(k.z - laser->laser_function(t));
}

PiFlux::momentum_type::momentum_type(h_float x, h_float y, h_float z) noexcept
    : cos_x(std::cos(x)), cos_y(std::cos(y)), cos_z(std::cos(z)), z(z) {}

void PiFlux::momentum_type::update(h_float x, h_float y, h_float z) noexcept {
    this->cos_x = std::cos(x);
    this->cos_y = std::cos(y);
    this->cos_z = std::cos(z);
    this->z = z;
}

void PiFlux::momentum_type::update_x(h_float val) noexcept {
    this->cos_x = std::cos(val);
}

void PiFlux::momentum_type::update_y(h_float val) noexcept {
    this->cos_y = std::cos(val);
}

void PiFlux::momentum_type::update_z(h_float val) noexcept {
    this->cos_z = std::cos(val);
    this->z = val;
}

void PiFlux::__time_evolution__(nd_vector& rhos,
                                Laser::Laser const* const laser,
                                const momentum_type& k,
                                const TimeIntegrationConfig& time_config) const {
    if (inverse_diagonal_relaxation_time > h_float{}) {
        return time_evolution_diagonal_relaxation(rhos, laser, k, time_config);
    }
    return time_evolution_sigma(rhos, laser, k, time_config);
}

h_float PiFlux::ic_sigma_x(const momentum_type& k, h_float alpha_beta_diff) const noexcept {
    const h_float epsilon = dispersion(k);
    if (is_zero(epsilon))
        return 0.;

    return alpha_beta_diff * k.cos_x / epsilon;
}

h_float PiFlux::ic_sigma_y(const momentum_type& k, h_float alpha_beta_diff) const noexcept {
    const h_float epsilon = dispersion(k);
    if (is_zero(epsilon))
        return 0.;

    return alpha_beta_diff * k.cos_y / epsilon;
}

h_float PiFlux::ic_sigma_z(const momentum_type& k, h_float alpha_beta_diff) const noexcept {
    const h_float epsilon = dispersion(k);
    if (is_zero(epsilon))
        return 0.;

    return alpha_beta_diff * k.cos_z / epsilon;
}

PiFlux::sigma_state_type PiFlux::ic_sigma(const momentum_type& k, h_float alpha_beta_diff) const noexcept {
    return sigma_state_type{ic_sigma_x(k, alpha_beta_diff),
                            ic_sigma_y(k, alpha_beta_diff),
                            ic_sigma_z(k, alpha_beta_diff)};
}

std::vector<h_float> PiFlux::current_density_lattice_sum(
    Laser::Laser const* const laser,
    TimeIntegrationConfig const& time_config,
    [[maybe_unused]] const int rank,
    [[maybe_unused]] const int n_ranks,
    const int n_z) const
{
    nd_vector rhos_buffer = nd_vector::Zero(time_config.n_measurements + 1);
    std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});

    const h_float time_step = time_config.measure_every();
    const int n_xy = n_z / 2;       // points in full [-π/2, π/2) for x and y
    const int n_half = n_xy / 2;    // interior points in (0, π/2); total in [0, π/2] is n_half+1
    const h_float dk = pi / n_xy;   // uniform spacing in x and y  (= 2π/n_z)

#ifdef NO_MPI
    std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for firstprivate(rhos_buffer) reduction(vec_plus : current_density_time) schedule(dynamic)
    for (int z = 0; z < n_z; ++z)
#else
    int jobs_per_rank = n_z / n_ranks;
    if (jobs_per_rank * n_ranks < n_z)
        ++jobs_per_rank;
    const int this_rank_min_z = rank * jobs_per_rank;
    const int this_rank_max_z = std::min(this_rank_min_z + jobs_per_rank, n_z);
    for (int z = this_rank_min_z; z < this_rank_max_z; ++z)
#endif
    {
        PROGRESS_BAR_UPDATE(n_z);

        const h_float kz = -pi + z * (2.0 * pi / n_z);
        momentum_type k;
        k.update_z(kz);

        nd_vector x_buffer = nd_vector::Zero(time_config.n_measurements + 1);

        // Sum over the first quadrant [0, π/2] × [0, π/2],
        // exploiting σ^z(kx,ky,kz) = σ^z(-kx,ky,kz) = σ^z(kx,-ky,kz).
        // Weight: 1 at the axes/boundaries, 2 in the interior (each direction).
        for (int ix = 0; ix <= n_half; ++ix) {
            const h_float kx = ix * dk;   // 0, π/n_xy, 2π/n_xy, …, π/2
            k.update_x(kx);
            const h_float wx = (ix == 0 || ix == n_half) ? 1.0 : 2.0;

            nd_vector y_buffer = nd_vector::Zero(time_config.n_measurements + 1);

            for (int iy = 0; iy <= n_half; ++iy) {
                const h_float ky = iy * dk;   // 0, π/n_xy, 2π/n_xy, …, π/2
                k.update_y(ky);
                const h_float wy = (iy == 0 || iy == n_half) ? 1.0 : 2.0;

                __time_evolution__(rhos_buffer, laser, k, time_config);

                y_buffer += wy * rhos_buffer;
            }

            x_buffer += wx * y_buffer;
        }

        // Multiply by the z-component of the current-density kernel.
        for (int i = 0; i <= time_config.n_measurements; ++i) {
            x_buffer[i] *= std::sin(kz - laser->laser_function(time_config.t_begin + i * time_step));
        }

        std::transform(current_density_time.begin(), current_density_time.end(),
                       x_buffer.begin(), current_density_time.begin(), std::plus<>());
    }
    std::cout << std::endl;

    // Normalise by the total number of k-points in the full BZ.
    const h_float n_points = static_cast<h_float>(n_z) * n_xy * n_xy;
    for (auto& j : current_density_time) {
        j /= n_points;
    }

    return current_density_time;
}

nd_vector PiFlux::improved_xy_integral(
    momentum_type& k,
    nd_vector& rhos_buffer,
    Laser::Laser const* const laser,
    TimeIntegrationConfig const& time_config) const
{
    constexpr h_float edge = 0.35 * pi;

    nd_vector x_buffer =
        nd_vector::Zero(time_config.n_measurements + 1);

#ifdef INTEGRATION_ERROR
    // Conservative estimate: sum of absolute regional differences.
    nd_vector error_buffer =
        nd_vector::Zero(time_config.n_measurements + 1);
#endif

    const auto transform = [](h_float x, h_float low, h_float high) {
        return 0.5 * (high - low) * x + 0.5 * (high + low);
    };

    const auto jacobian = [](h_float low, h_float high) {
        return 0.5 * (high - low);
    };

    // Integrate one rectangular region.
    // NX and NY specify half the number of nodes in the main rule.
    auto integrate_region = [&]<int NX, int NY>(
        h_float x_low,
        h_float x_high,
        h_float y_low,
        h_float y_high)
    {
        const h_float transform_weight =
            jacobian(x_low, x_high) * jacobian(y_low, y_high);

        // LowOrder=false: 2*NX by 2*NY nodes.
        // LowOrder=true:     NX by NY nodes.
        // Each rule uses its own Gauss-Legendre nodes and weights.
        auto integrate_rule = [&]<bool LowOrder>() -> nd_vector {
            using x_rule = std::conditional_t<
                LowOrder,
                gauss::container<NX>,
                gauss::container<2 * NX>>;

            using y_rule = std::conditional_t<
                LowOrder,
                gauss::container<NY>,
                gauss::container<2 * NY>>;

            constexpr int n_x = LowOrder ? NX / 2 : NX;
            constexpr int n_y = LowOrder ? NY / 2 : NY;

            nd_vector result =
                nd_vector::Zero(time_config.n_measurements + 1);

            for (int ix = 0; ix < n_x; ++ix) {
                const h_float x_node = x_rule::abscissa[ix];
                const h_float x_weight = x_rule::weights[ix];

                for (int iy = 0; iy < n_y; ++iy) {
                    const h_float y_node = y_rule::abscissa[iy];
                    const h_float y_weight = y_rule::weights[iy];

                    const h_float quadrature_weight =
                        transform_weight * x_weight * y_weight;

                    // Use all four sign combinations of the positive
                    // Gauss-Legendre nodes.
                    for (int sx : {-1, 1}) {
                        k.update_x(transform(
                            sx * x_node, x_low, x_high));

                        for (int sy : {-1, 1}) {
                            k.update_y(transform(
                                sy * y_node, y_low, y_high));

                            __time_evolution__(
                                rhos_buffer, laser, k, time_config);

                            result += quadrature_weight * rhos_buffer;
                        }
                    }
                }
            }

            return result;
        };

        const nd_vector high_order = integrate_rule.template operator()<false>();
        x_buffer += high_order;

#ifdef INTEGRATION_ERROR
        const nd_vector low_order = integrate_rule.template operator()<true>();

        // Accumulate regional errors without allowing cancellation between
        // different subregions.
        error_buffer += (high_order - low_order).cwiseAbs();
#endif
    };

    // 1. Central region: high resolution in both dimensions.
    integrate_region.template operator()<N_fine, N_fine / 2>(
        edge, pi - edge,
        edge, 0.5 * pi);

    // 2. Lower strip: increased x resolution, coarser y resolution.
    integrate_region.template operator()<3 * N_coarse, N_coarse>(
        0.0, pi,
        0.0, edge);

    // 3. Left side region.
    integrate_region.template operator()<N_coarse, N_coarse>(
        0.0, edge,
        edge, 0.5 * pi);

    // 4. Right side region.
    integrate_region.template operator()<N_coarse, N_coarse>(
        pi - edge, pi,
        edge, 0.5 * pi);

#ifdef INTEGRATION_ERROR
    std::cout
        << "Estimated XY quadrature error (max over time) = "
        << error_buffer.maxCoeff()
        << '\n';
#endif

    return x_buffer;
}

std::vector<h_float> PiFlux::current_density_continuum_limit(
    Laser::Laser const* const laser,
    TimeIntegrationConfig const& time_config,
    [[maybe_unused]] const int rank,
    [[maybe_unused]] const int n_ranks,
    [[maybe_unused]] const int n_z) const
{
    typedef gauss::container<2 * z_range> z_gauss;

    nd_vector rhos_buffer;
    nd_vector x_buffer;

    std::vector<h_float> current_density_time(
        time_config.n_measurements + 1, h_float{});

    const h_float time_step = time_config.measure_every();
    constexpr h_float z_jacobian = pi;

#ifdef NO_MPI
    std::vector<int> progresses(omp_get_max_threads(), int{});

#pragma omp parallel for private(rhos_buffer, x_buffer) \
    reduction(vec_plus : current_density_time)
    for (int z = 0; z < z_range; ++z)
#else
    int jobs_per_rank = z_range / n_ranks;

    if (rank == 0) {
        std::cout << "Jobs per rank: " << jobs_per_rank
                  << "\nn_ranks: " << n_ranks
                  << "\nz_range: " << z_range
                  << std::endl;
    }

    if (jobs_per_rank * n_ranks < z_range)
        ++jobs_per_rank;

    const int this_rank_min_z = rank * jobs_per_rank;
    const int this_rank_max_z = std::min(this_rank_min_z + jobs_per_rank, z_range);

    for (int z = this_rank_min_z; z < this_rank_max_z; ++z)
#endif
    {
        PROGRESS_BAR_UPDATE(z_range);

        momentum_type k;

        // Positive Gauss–Legendre node.
        k.update_z(pi * z_gauss::abscissa[z]);

        x_buffer = improved_xy_integral(
            k, rhos_buffer, laser, time_config);

        for (int i = 0; i <= time_config.n_measurements; ++i) {
            x_buffer[i] *= z_jacobian * z_gauss::weights[z]
                * std::sin(k.z - laser->laser_function( time_config.t_begin + i * time_step));
        }

        std::transform(
            current_density_time.begin(),
            current_density_time.end(),
            x_buffer.begin(),
            current_density_time.begin(),
            std::plus<>());

        // Corresponding negative Gauss–Legendre node.
        k.update_z(-pi * z_gauss::abscissa[z]);

        x_buffer = improved_xy_integral(
            k, rhos_buffer, laser, time_config);

        for (int i = 0; i <= time_config.n_measurements; ++i) {
            x_buffer[i] *= z_jacobian * z_gauss::weights[z]
                * std::sin(k.z - laser->laser_function(time_config.t_begin + i * time_step));
        }

        std::transform( current_density_time.begin(), current_density_time.end(), 
            x_buffer.begin(), current_density_time.begin(), std::plus<>());
    }

    return current_density_time;
}

std::vector<OccupationContainer> PiFlux::compute_occupation_numbers(Laser::Laser const* const laser,
                                                                    TimeIntegrationConfig const& time_config,
                                                                    const int N) const {
    const h_float dx = pi / N;
    const h_float dz = pi / N;

    auto k_xy = [dx](int x) { return x * dx; };
    auto k_z = [dz](int z) { return dz * z; };

    auto coordinate_shift = [N, dz](int z, h_float laser_value) -> int {
        const int shift = static_cast<int>(std::round(laser_value / dz));
        return (((z - shift) % N) + N) % N;
    };

    const h_float measure_every = time_config.measure_every();
    std::vector<OccupationContainer::occupation_t> occ_buffer(time_config.n_measurements + 1);
    std::vector<OccupationContainer> computed_occupations(time_config.n_measurements + 1, OccupationContainer(N));

    std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for firstprivate(occ_buffer)
    for (int i = 0; i < N * N; ++i) {
        PROGRESS_BAR_UPDATE(N * N);

        const int x = i / N;
        const int z = i % N;
        momentum_type k(k_xy(x), 0.5 * pi, k_z(z));
        if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
            k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
            k.cos_z *= (1. - 0.5 * RESCUE_TRAFO * RESCUE_TRAFO);
        }

        evolve_occupation_numbers(occ_buffer, laser, k, time_config);

        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;
        for (int i = 0; i <= time_config.n_measurements; ++i) {
            computed_occupations[i](x, coordinate_shift(z, laser->laser_function(t_begin))) = occ_buffer[i];

            t_begin = t_end;
            t_end += measure_every;
        }
    }
    return computed_occupations;
}

std::array<std::vector<h_float>, 2> PiFlux::current_per_energy(Laser::Laser const* const laser,
                                                               TimeIntegrationConfig const& time_config,
                                                               const int N) const {
    constexpr h_float energy_cut = 0.5 * sqrt_3;

    const h_float dxy = pi / N;
    const h_float dz = pi / N;

    auto k_xy = [dxy](int x) { return x * dxy; };
    auto k_z = [dz](int z) { return dz * z; };

    const h_float time_step = time_config.measure_every();

    nd_vector rhos_buffer;

    std::vector<h_float> current_density_non_dirac(time_config.n_measurements + 1, h_float{});
    std::vector<h_float> current_density_dirac(time_config.n_measurements + 1, h_float{});

#ifndef MROCK_CL1_CASCADE
    std::vector<int> progresses(omp_get_max_threads(), int{});
#endif
#pragma omp parallel for private(rhos_buffer) reduction(vec_plus : current_density_dirac, current_density_non_dirac) \
    schedule(dynamic)
    for (int i = 0; i < N * N * N; ++i) {
        PROGRESS_BAR_UPDATE(N * N * N);

        const int z = i / (N * N);
        const int y = (i / N) % N;
        const int x = i % N;

        momentum_type k(k_xy(x), k_xy(y), k_z(z));
        if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
            k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
            k.cos_z *= (1. - 0.5 * RESCUE_TRAFO * RESCUE_TRAFO);
        }

        if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
            k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
            k.cos_z *= (1. - 0.5 * RESCUE_TRAFO * RESCUE_TRAFO);
        }
        __time_evolution__(rhos_buffer, laser, k, time_config);

        for (int i = 0; i <= time_config.n_measurements; ++i) {
            const h_float laser_value = laser->laser_function(i * time_step + time_config.t_begin);
            const h_float instantaneous_energy = dispersion(k.shift_z(laser_value));

            if (instantaneous_energy < energy_cut) {
                current_density_dirac[i] += rhos_buffer(i) * std::sin(k.z - laser_value);
            } else {
                current_density_non_dirac[i] += rhos_buffer(i) * std::sin(k.z - laser_value);
            }
        }

        /*
         *  -z
         */
        k.invert();
        if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
            k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
            k.cos_z *= (1. - 0.5 * RESCUE_TRAFO * RESCUE_TRAFO);
        }
        __time_evolution__(rhos_buffer, laser, k, time_config);
        for (int i = 0; i <= time_config.n_measurements; ++i) {
            const h_float laser_value = laser->laser_function(i * time_step + time_config.t_begin);
            const h_float instantaneous_energy = dispersion(k.shift_z(laser_value));

            if (instantaneous_energy < energy_cut) {
                current_density_dirac[i] += rhos_buffer(i) * std::sin(k.z - laser_value);
            } else {
                current_density_non_dirac[i] += rhos_buffer(i) * std::sin(k.z - laser_value);
            }
        }
    }

    for (auto& val : current_density_dirac) {
        val /= N * N * N;
    }
    for (auto& val : current_density_non_dirac) {
        val /= N * N * N;
    }
    return {current_density_dirac, current_density_non_dirac};
}

std::array<h_float, 3> PiFlux::diagonal_sigma(sigma_state_type const& input, momentum_type const& k) const {
    const h_float z = k.cos_z + dispersion(k);
    const h_float norm = 2 * dispersion(k) * z;
    const h_float __xz = 2 * k.cos_x * z;
    const h_float __yz = 2 * k.cos_y * z;
    const h_float __xy = 2 * k.cos_x * k.cos_y;

    return {(input[0] * (z * z - k.cos_x * k.cos_x + k.cos_y * k.cos_y) - __xy * input[1] - __xz * input[2]) / norm,
            (input[1] * (z * z + k.cos_x * k.cos_x - k.cos_y * k.cos_y) - __xy * input[0] - __yz * input[2]) / norm,
            (input[2] * (z * z - k.cos_x * k.cos_x - k.cos_y * k.cos_y) + __xz * input[0] + __yz * input[1]) / norm};
}
}  // namespace HHG::Systems