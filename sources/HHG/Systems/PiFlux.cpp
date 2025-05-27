#include "PiFlux.hpp"
#include "../GeneralMagnus.hpp"
#include "../Laser/gauss.hpp"
#include "../thread_gauss.hpp"

#include <cmath>
#include <cassert>
#include <numeric>
#include <random>
#include <omp.h>
#include <nlohmann/json.hpp>

#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/progress_bar.hpp>

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;

typedef Eigen::Vector<HHG::h_float, 3> sigma_state_type;
typedef runge_kutta_fehlberg78<sigma_state_type> sigma_error_stepper_type;

constexpr HHG::h_float abs_error = 1.0e-12;
constexpr HHG::h_float rel_error = 1.0e-8;

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size(), decltype(omp_orig)::value_type{}))

#ifdef NO_MPI
#define PROGRESS_BAR_UPDATE(z_max) ++(progresses[omp_get_thread_num()]); \
            if (omp_get_thread_num() == 0) { \
                mrock::utility::progress_bar( \
                    static_cast<float>(std::reduce(progresses.begin(), progresses.end())) / static_cast<float>((z_max)) \
                ); \
            }
#else
#define PROGRESS_BAR_UPDATE(z_max)
#endif

//#define INTEGRATION_ERROR
#ifdef INTEGRATION_ERROR
#define INTEGRATOR_TYPEDEF(N) using __gauss = gauss::container<2 * (N)>; \
                          using __error = gauss::container<(N)>;
#define ERROR_INTEGRATOR_WEIGHT h_float(!(i&1)) * transform_weight * __error::weights[i / 2]
#else
#define INTEGRATOR_TYPEDEF(N) using __gauss = gauss::container<2 * (N)>;
#define ERROR_INTEGRATOR_WEIGHT h_float{}
#endif

//#define DEBUG_INTEGRATE

namespace HHG::Systems {
    PiFlux::PiFlux(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _decay_time)
        : beta(is_zero(temperature) ? std::numeric_limits<h_float>::infinity() : _photon_energy / (k_B * temperature)), 
            E_F(_E_F / _photon_energy), 
            hopping_element(_band_width / sqrt_12), 
            lattice_constant(sqrt_3 * hbar * _v_F / (_photon_energy * _band_width)),
            inverse_decay_time((1e15 * hbar) / (_decay_time * _photon_energy))
    {
        //gauss::precompute<40>();
        //gauss::precompute<80>();
        //gauss::precompute<120>();
        //gauss::precompute<160>();
        //gauss::precompute<240>();
        //gauss::precompute<480>();
        //abort();
    }

    void PiFlux::time_evolution_magnus(nd_vector &rhos, Laser::Laser const *const laser, const momentum_type &k, const TimeIntegrationConfig &time_config) const
    {
        const h_float alpha_0 = fermi_function(E_F + dispersion(k), beta);
        const h_float beta_0 = fermi_function(E_F - dispersion(k), beta);

        sigma_state_type current_state = { 2. * alpha_0 * beta_0, h_float{0}, alpha_0 * alpha_0 - beta_0 * beta_0 };

        const h_float measure_every = time_config.measure_every();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.conservativeResize(time_config.n_measurements + 1);
        rhos[0] = current_state(2);

        GeneralMagnus magnus;
        std::array<std::array<h_float, 3>, 4> expansion_coefficients;
        for (int i = 1; i <= time_config.n_measurements; ++i) {
            expansion_coefficients = this->magnus_coefficients(k, measure_every, t_begin, laser);
            current_state.applyOnTheLeft(magnus.Omega(expansion_coefficients[0], expansion_coefficients[1], expansion_coefficients[2], expansion_coefficients[3]));

            rhos[i] = current_state(2);
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void PiFlux::time_evolution_sigma(nd_vector &rhos, Laser::Laser const *const laser, const momentum_type &k, const TimeIntegrationConfig &time_config) const
    {
        const h_float prefactor = 4 * hopping_element;

        const h_float alpha2 = occupation_a(k);
        const h_float beta2 = occupation_b(k);
        const h_float alpha_beta_diff = alpha2 - beta2;
        const h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);
        const h_float z_epsilon = k.cos_z + dispersion(k);

        sigma_state_type current_state = { ic_sigma_x(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_y(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_z(k, alpha_beta_diff, alpha_beta_prod, z_epsilon) };

        auto right_side = [this, &k, &laser, &prefactor](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const sigma_state_type m = {k.cos_x, k.cos_y, std::cos(k.z - laser->laser_function(t))};
            dxdt = prefactor * m.cross(state);
        };

        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.conservativeResize(time_config.n_measurements + 1);
        rhos[0] = current_state(2);// * std::sin(k.z - laser->laser_function(t_begin));

        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            rhos[i] = current_state(2);// * std::sin(k.z - laser->laser_function(t_end));
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void PiFlux::time_evolution_decay(nd_vector &rhos, Laser::Laser const *const laser, const momentum_type &k, const TimeIntegrationConfig &time_config) const
    {
        assert(!is_zero(dispersion(k)));
        const h_float prefactor = 4 * hopping_element;

        const h_float alpha2 = occupation_a(k);
        const h_float beta2 = occupation_b(k);
        const h_float alpha_beta_diff = alpha2 - beta2;
        const h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);
        const h_float z_epsilon = k.cos_z + dispersion(k);

        sigma_state_type current_state = { ic_sigma_x(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_y(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_z(k, alpha_beta_diff, alpha_beta_prod, z_epsilon) };
        const sigma_state_type initial_state = current_state;

        auto right_side = [this, &k, &laser, &prefactor, &initial_state](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const sigma_state_type m = {k.cos_x, k.cos_y, std::cos(k.z - laser->laser_function(t))};
            dxdt = prefactor * m.cross(state) - inverse_decay_time * (state - initial_state);
        };

        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.resize(time_config.n_measurements + 1);
        rhos[0] = current_state(2);// * std::sin(k.z - laser->laser_function(t_begin));

        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            rhos[i] = current_state(2);// * std::sin(k.z - laser->laser_function(t_end));
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    std::array<std::vector<h_float>, n_debug_points> PiFlux::compute_current_density_debug(Laser::Laser const * const laser, 
        TimeIntegrationConfig const& time_config, const int n_z) const
    {
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
            picked[i] = 0.5 * pi - (i * 0.02 * pi);//(i + 1) * 0.5 * pi / (n_debug_points);
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
            std::cout << "#" << i << "  k_y=" << picked[i] << ":    " << (error_evolutions[i] - time_evolutions[i]).norm() << std::endl;
#else
            momentum_type k(picked_x, picked[i], picked_z);
            __time_evolution__(time_evolutions[i], laser, k, time_config);
#endif
        }
        // end debug setup

        std::array<std::vector<h_float>, n_debug_points> time_evolutions_std;
        for(int i = 0; i < n_debug_points; ++i) {
            time_evolutions_std[i].resize(time_config.n_measurements + 1);
            std::copy(time_evolutions[i].begin(), time_evolutions[i].end(), time_evolutions_std[i].begin());
        }
        return time_evolutions_std;
    }

    std::vector<h_float> PiFlux::compute_current_density(Laser::Laser const *const laser, TimeIntegrationConfig const &time_config, const int rank, const int n_ranks, const int n_z) const
    {
        //return current_density_lattice_sum(laser, time_config, rank, n_ranks, n_z);
        return current_density_continuum_limit(laser, time_config, rank, n_ranks, n_z);
        //return current_density_monte_carlo(laser, time_config, rank, n_ranks, n_z);
    }

    std::string PiFlux::info() const
    {
        return  "PiFlux\nT=" + std::to_string(1.0 / beta) + "\n" 
                + "E_F=" + std::to_string(E_F) + "\n" 
                + "t=" + std::to_string(hopping_element) + "\n" 
                + "d=" + std::to_string(lattice_constant) + "\n";
    }

    h_float PiFlux::dispersion(const momentum_type& k) const
    {
        return sqrt(k.cos_x*k.cos_x + k.cos_y*k.cos_y + k.cos_z*k.cos_z);
    }

    h_float PiFlux::occupation_a(const momentum_type& k) const 
    {
        return fermi_function(E_F + 2 * hopping_element * dispersion(k), beta);
    }

    h_float PiFlux::occupation_b(const momentum_type& k) const
    {
        return fermi_function(E_F - 2 * hopping_element * dispersion(k), beta);
    }

    h_float PiFlux::alpha(const momentum_type &k, h_float t, Laser::Laser const *const laser) const
    {
        return k.cos_z - std::cos(k.z - laser->laser_function(t));
    }

    h_float PiFlux::xi(const momentum_type &k, h_float t, Laser::Laser const * const laser) const
    {
        return k.cos_x*k.cos_x + k.cos_y*k.cos_y + k.cos_z*std::cos(k.z - laser->laser_function(t));
    }

    std::array<std::array<h_float, 3>, 4> PiFlux::magnus_coefficients(const momentum_type& k, h_float delta_t, h_float t_0, Laser::Laser const * const laser) const
    {
        using namespace Laser;
        assert(!k.is_dirac_point());
        const h_float prefactor = 4. * hopping_element * delta_t / dispersion(k);

        std::array<std::array<h_float, 3>, 4> coeffs;
        h_float current_alpha, current_xi;

        for (int i = 0; i < n_gauss; ++i) {
            current_alpha = this->alpha(k, t_0 + delta_t * abscissa[i], laser);
            current_xi = this->xi(k, t_0 + delta_t * abscissa[i], laser);

            coeffs[0][0] += weights[i] * k.cos_x * current_alpha;
            coeffs[1][0] += weights[i] * legendre_2[i] * k.cos_x * current_alpha;
            coeffs[2][0] += weights[i] * legendre_3[i] * k.cos_x * current_alpha;
            coeffs[3][0] += weights[i] * legendre_4[i] * k.cos_x * current_alpha;

            coeffs[0][1] += weights[i] * k.cos_y * current_alpha;
            coeffs[1][1] += weights[i] * legendre_2[i] * k.cos_y * current_alpha;
            coeffs[2][1] += weights[i] * legendre_3[i] * k.cos_y * current_alpha;
            coeffs[3][1] += weights[i] * legendre_4[i] * k.cos_y * current_alpha;

            coeffs[0][2] += weights[i] * current_xi;
            coeffs[1][2] += weights[i] * legendre_2[i] * current_xi;
            coeffs[2][2] += weights[i] * legendre_3[i] * current_xi;
            coeffs[3][2] += weights[i] * legendre_4[i] * current_xi;
        }

        for (int i = 0; i < 4; ++i) {
            for (auto& coeff : coeffs[i]) {
                coeff *= (2 * i + 1) * prefactor;
            }
        }
        return coeffs;
    }

    PiFlux::momentum_type::momentum_type(h_float x, h_float y, h_float z) noexcept
        : cos_x(std::cos(x)), cos_y(std::cos(y)), cos_z(std::cos(z)), z(z)
    { }

    PiFlux::momentum_type PiFlux::momentum_type::SymmetrizedRandom()
    {
        thread_local static std::mt19937 gen([] {
            std::random_device dev;
            return std::mt19937(dev());
        }());
        static std::uniform_real_distribution<h_float> dist_z(0.0, pi);
        static std::uniform_real_distribution<h_float> dist_xy(0.0, 0.5 * pi);

        return momentum_type(dist_xy(gen), dist_xy(gen), dist_z(gen));
    }

    void PiFlux::momentum_type::update(h_float x, h_float y, h_float z) noexcept
    {
        this->cos_x = std::cos(x);
        this->cos_y = std::cos(y);
        this->cos_z = std::cos(z);
        this->z = z;
    }

    void PiFlux::momentum_type::update_x(h_float val) noexcept
    {
        this->cos_x = std::cos(val);
    }

    void PiFlux::momentum_type::update_y(h_float val) noexcept
    {
        this->cos_y = std::cos(val);
    }

    void PiFlux::momentum_type::update_z(h_float val) noexcept
    {
        this->cos_z = std::cos(val);
        this->z = val;
    }

    void PiFlux::__time_evolution__(nd_vector& rhos, Laser::Laser const * const laser, 
        const momentum_type& k, const TimeIntegrationConfig& time_config) const
    {
        if (inverse_decay_time > h_float{}) {
            return time_evolution_decay(rhos, laser, k, time_config);
        }
        return time_evolution_sigma(rhos, laser, k, time_config);
    }

    h_float PiFlux::ic_sigma_x(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept
    {
        return (alpha_beta_diff * k.cos_x * z_epsilon + alpha_beta_prod * (k.cos_y * k.cos_y + k.cos_z * z_epsilon)) / (dispersion(k) * z_epsilon);
    }

    h_float PiFlux::ic_sigma_y(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept
    {
        return (alpha_beta_diff * k.cos_y * z_epsilon - alpha_beta_prod * k.cos_x * k.cos_y) / (dispersion(k) * z_epsilon);
    }

    h_float PiFlux::ic_sigma_z(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept
    {
        return (alpha_beta_diff * k.cos_z * z_epsilon - alpha_beta_prod * k.cos_x * z_epsilon) / (dispersion(k) * z_epsilon);
    }

    std::vector<h_float> PiFlux::current_density_lattice_sum(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
        const int rank, const int n_ranks, const int n_z) const
    {
        nd_vector rhos_buffer = nd_vector::Zero(time_config.n_measurements + 1);
        std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});

        const h_float momentum_ratio = 2.0 * pi / n_z;
        const int n_xy = n_z / 2;

        const h_float time_step = time_config.measure_every();

#ifdef NO_MPI
        std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for firstprivate(rhos_buffer) reduction(vec_plus:current_density_time) schedule(dynamic)
        for (int z = 0; z < n_z; ++z)
#else
        int jobs_per_rank = n_z / n_ranks;
        if (jobs_per_rank * n_ranks < n_z) ++jobs_per_rank;
        const int this_rank_min_z = rank * jobs_per_rank + 1;
        const int this_rank_max_z = this_rank_min_z + jobs_per_rank > n_z ? n_z : this_rank_min_z + jobs_per_rank;
        for (int z = this_rank_min_z; z < this_rank_max_z; ++z)
#endif
        {
            PROGRESS_BAR_UPDATE(n_z);

            momentum_type k;
            k.update_z(z * momentum_ratio - pi);

            nd_vector x_buffer = nd_vector::Zero(time_config.n_measurements + 1); 
            // At k_x = k_y = \pm pi (that would be x=y=\pm n_z / 2), we have nothing but a rotation around the z-axis, 
            // therefore sigma_z = const.
            // We also have the symmetry that rho(k_x) = rho(-k_x) as everything depends merely on cos(k_x).
            // The same applies to k_y, but not to k_z.
            for (int x = 1; x < n_xy / 2; ++x) {
                k.update_x(x * momentum_ratio);

                nd_vector y_buffer = nd_vector::Zero(time_config.n_measurements + 1);
                for (int y = 1; y < n_xy / 2; ++y) {
                    k.update_y(y * 0.5 * momentum_ratio);
                    __time_evolution__(rhos_buffer, laser, k, time_config);
                    y_buffer += rhos_buffer;
                }
                y_buffer *= 2.0;

                k.update_y(0.0);
                __time_evolution__(rhos_buffer, laser, k, time_config);
                y_buffer += rhos_buffer;

                k.update_y(pi / 2.0);
                __time_evolution__(rhos_buffer, laser, k, time_config);
                y_buffer += rhos_buffer;

                x_buffer += y_buffer;
            }
            x_buffer *= 2.0;

            {
                k.update_x(0.0);

                nd_vector y_buffer = nd_vector::Zero(time_config.n_measurements + 1);
                for (int y = 1; y < n_xy / 2; ++y) {
                    k.update_y(y * 0.5 * momentum_ratio);
                    __time_evolution__(rhos_buffer, laser, k, time_config);
                    y_buffer += rhos_buffer;
                }
                y_buffer *= 2.0;

                k.update_y(0.0);
                __time_evolution__(rhos_buffer, laser, k, time_config);
                y_buffer += rhos_buffer;

                k.update_y(pi / 2.0);
                __time_evolution__(rhos_buffer, laser, k, time_config);
                y_buffer += rhos_buffer;

                x_buffer += y_buffer;
            }
            {
                k.update_x(pi / 2.0);

                nd_vector y_buffer = nd_vector::Zero(time_config.n_measurements + 1);
                for (int y = 1; y < n_xy / 2; ++y) {
                    k.update_y(y * 0.5 * momentum_ratio);
                    __time_evolution__(rhos_buffer, laser, k, time_config);
                    y_buffer += rhos_buffer;
                }
                y_buffer *= 2.0;

                k.update_y(0.0);
                __time_evolution__(rhos_buffer, laser, k, time_config);
                y_buffer += rhos_buffer;

                x_buffer += y_buffer;
            }

            for (int i = 0; i <= time_config.n_measurements; ++i) {
                x_buffer[i] *= std::sin(k.z - laser->laser_function(i * time_step));
            }

            std::transform(current_density_time.begin(), current_density_time.end(), x_buffer.begin(), current_density_time.begin(), std::plus<>());
        }
        std::cout << std::endl;

        for (auto& j : current_density_time) {
            j /= (n_z * n_z * n_z);
        }

        return current_density_time;
    }

    nd_vector PiFlux::xy_integral(momentum_type& k, nd_vector& rhos_buffer, Laser::Laser const * const laser, TimeIntegrationConfig const& time_config) const {
        typedef gauss::container<2 * n_xy_inner> xy_inner;

        nd_vector x_buffer = nd_vector::Zero(time_config.n_measurements + 1);

        for (int i = 0; i < n_xy_inner; ++i) {
            k.update_x(0.5 * pi * xy_inner::abscissa[i]);

            for (int j = 0; j < n_xy_inner; ++j) {
                k.update_y(0.5 * pi * xy_inner::abscissa[j]);
                __time_evolution__(rhos_buffer, laser, k, time_config);

                x_buffer += xy_inner::weights[i] * xy_inner::weights[j] * rhos_buffer;
            }
        }
        return x_buffer;
    }

    nd_vector PiFlux::improved_xy_integral(momentum_type& k, nd_vector& rhos_buffer, Laser::Laser const * const laser, TimeIntegrationConfig const& time_config) const {
        constexpr h_float edge = 0.35 * pi;
        
        nd_vector x_buffer = nd_vector::Zero(time_config.n_measurements + 1);
#ifdef INTEGRATION_ERROR
        nd_vector error_buffer = nd_vector::Zero(time_config.n_measurements + 1);
#endif
        auto transform = [](h_float x, h_float low, h_float high) {
            return 0.5 * (high - low) * x + 0.5 * (high + low);
        };
        auto weight = [](h_float low, h_float high) {
            return 0.5 * (high - low);
        };

        auto y_integration = [&]<int __N>(h_float y_low, h_float y_high, h_float main_weight, h_float error_weight) {
            INTEGRATOR_TYPEDEF(__N);

            for (int j = 0; j < __N; ++j) {
                k.update_y(transform(__gauss::abscissa[j], y_low, y_high));
                __time_evolution__(rhos_buffer, laser, k, time_config);
                x_buffer += main_weight * __gauss::weights[j] * rhos_buffer;

#ifdef INTEGRATION_ERROR
                if (!((j&1) || is_zero(error_weight))) 
                    error_buffer += error_weight * __error::weights[j / 2] * rhos_buffer;
#endif

                k.update_y(transform(-__gauss::abscissa[j], y_low, y_high));
                __time_evolution__(rhos_buffer, laser, k, time_config);
                x_buffer += main_weight * __gauss::weights[j] * rhos_buffer;

#ifdef INTEGRATION_ERROR
                if (!((j&1) || is_zero(error_weight))) 
                    error_buffer += error_weight * __error::weights[j / 2] * rhos_buffer;
#endif
            }
        };

        {
            constexpr h_float x_low = edge;
            constexpr h_float x_high = pi - edge;
            constexpr h_float y_low = edge;
            constexpr h_float y_high = 0.5 * pi;
            constexpr h_float transform_weight = weight(x_low, x_high) * weight(y_low, y_high);

            INTEGRATOR_TYPEDEF(N_fine);
            for (int i = 0; i < N_fine; ++i) {
                k.update_x(transform(__gauss::abscissa[i], x_low, x_high));
                y_integration.template operator()<N_fine / 2>(y_low, y_high, __gauss::weights[i] * transform_weight, 
                    ERROR_INTEGRATOR_WEIGHT);
                k.update_x(transform(-__gauss::abscissa[i], x_low, x_high));
                y_integration.template operator()<N_fine / 2>(y_low, y_high, __gauss::weights[i] * transform_weight, 
                    ERROR_INTEGRATOR_WEIGHT);
            }

            //std::cout << "#1 Error = " << (error_buffer - x_buffer).cwiseAbs().maxCoeff() << std::endl;
            //x_buffer.setZero();
            //error_buffer.setZero();
        }

        {
            constexpr h_float x_low = 0.0;
            constexpr h_float x_high = pi;
            constexpr h_float y_low = 0.0;
            constexpr h_float y_high = edge;
            constexpr h_float transform_weight = weight(x_low, x_high) * weight(y_low, y_high);

            constexpr int Nx = 3 * N_coarse;
            INTEGRATOR_TYPEDEF(Nx);
            for (int i = 0; i < Nx; ++i) {
                k.update_x(transform(__gauss::abscissa[i], x_low, x_high));
                y_integration.template operator()<N_coarse>(y_low, y_high, __gauss::weights[i] * transform_weight, 
                    ERROR_INTEGRATOR_WEIGHT);
                k.update_x(transform(-__gauss::abscissa[i], x_low, x_high));
                y_integration.template operator()<N_coarse>(y_low, y_high, __gauss::weights[i] * transform_weight, 
                    ERROR_INTEGRATOR_WEIGHT);
            }

            //std::cout << "#2 Error = " << (error_buffer - x_buffer).cwiseAbs().maxCoeff() << std::endl;
            //x_buffer.setZero();
            //error_buffer.setZero();
        }

        {
            constexpr h_float x_low = 0.0;
            constexpr h_float x_high = edge;
            constexpr h_float y_low = edge;
            constexpr h_float y_high = 0.5 * pi;
            constexpr h_float transform_weight = weight(x_low, x_high) * weight(y_low, y_high);

            INTEGRATOR_TYPEDEF(N_coarse);
            for (int i = 0; i < N_coarse; ++i) {
                k.update_x(transform(__gauss::abscissa[i], x_low, x_high));
                y_integration.template operator()<N_coarse>(y_low, y_high, __gauss::weights[i] * transform_weight, 
                    ERROR_INTEGRATOR_WEIGHT);
                k.update_x(transform(-__gauss::abscissa[i], x_low, x_high));
                y_integration.template operator()<N_coarse>(y_low, y_high, __gauss::weights[i] * transform_weight, 
                    ERROR_INTEGRATOR_WEIGHT);
            }

            //std::cout << "#3 Error = " << (error_buffer - x_buffer).cwiseAbs().maxCoeff() << std::endl;
            //x_buffer.setZero();
            //error_buffer.setZero();
        }

        {
            constexpr h_float x_low = pi - edge;
            constexpr h_float x_high = pi;
            constexpr h_float y_low = edge;
            constexpr h_float y_high = 0.5 * pi;
            constexpr h_float transform_weight = weight(x_low, x_high) * weight(y_low, y_high);

            INTEGRATOR_TYPEDEF(N_coarse);
            for (int i = 0; i < N_coarse; ++i) {
                k.update_x(transform(__gauss::abscissa[i], x_low, x_high));
                y_integration.template operator()<N_coarse>(y_low, y_high, __gauss::weights[i] * transform_weight, 
                    ERROR_INTEGRATOR_WEIGHT);
                k.update_x(transform(-__gauss::abscissa[i], x_low, x_high));
                y_integration.template operator()<N_coarse>(y_low, y_high, __gauss::weights[i] * transform_weight, 
                    ERROR_INTEGRATOR_WEIGHT);
            }
#ifdef INTEGRATION_ERROR
            std::cout << "#4 Error = " << (error_buffer - x_buffer).cwiseAbs().maxCoeff() << std::endl;
#endif
        }

        return x_buffer;
    }

    std::vector<h_float> PiFlux::current_density_continuum_limit(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
        const int rank, const int n_ranks, const int n_z) const
    {
        typedef gauss::container<2 * z_range> z_gauss;

        nd_vector rhos_buffer;
        nd_vector x_buffer;
        std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});

        const h_float time_step = time_config.measure_every();
#ifdef NO_MPI
        std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for private(rhos_buffer, x_buffer) reduction(vec_plus:current_density_time)
        for (int z = 0; z < z_range; ++z)
#else
        int jobs_per_rank = z_range / n_ranks;
        if (rank == 0) { std::cout << "Jobs per rank: " << jobs_per_rank << "\nn_ranks: " << n_ranks << "\nz_range: " << z_range << std::endl; }
        if (jobs_per_rank * n_ranks < z_range) ++jobs_per_rank;
        const int this_rank_min_z = rank * jobs_per_rank;
        const int this_rank_max_z = this_rank_min_z + jobs_per_rank > z_range ? z_range : this_rank_min_z + jobs_per_rank;
        for (int z = this_rank_min_z; z < this_rank_max_z; ++z)
#endif
        {
            PROGRESS_BAR_UPDATE(z_range);
            momentum_type k;
            k.update_z(pi * z_gauss::abscissa[z]);
            x_buffer = improved_xy_integral(k, rhos_buffer, laser, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                x_buffer[i] *= z_gauss::weights[z] * std::sin(k.z - laser->laser_function(i * time_step + time_config.t_begin));
            }
            std::transform(current_density_time.begin(), current_density_time.end(), x_buffer.begin(), current_density_time.begin(), std::plus<>());
        
            /*
            *  -z
            */
            k.update_z(-pi * z_gauss::abscissa[z]);
            x_buffer = improved_xy_integral(k, rhos_buffer, laser, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                x_buffer[i] *= z_gauss::weights[z] * std::sin(k.z - laser->laser_function(i * time_step + time_config.t_begin));
            }
            std::transform(current_density_time.begin(), current_density_time.end(), x_buffer.begin(), current_density_time.begin(), std::plus<>());
        }
        return current_density_time;
    }

    std::vector<h_float> PiFlux::current_density_monte_carlo(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
        const int rank, const int n_ranks, const int n_z) const
    {
        std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});
#ifdef NO_MPI
        nd_vector rhos_buffer = nd_vector::Zero(time_config.n_measurements + 1);
        const h_float time_step = time_config.measure_every();

        std::vector<int> progresses(omp_get_max_threads(), int{});
        std::vector<std::mt19937> gens;
        gens.reserve(omp_get_max_threads());
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            std::random_device dev;
            gens.emplace_back(std::mt19937(dev()));
        }
#pragma omp parallel for firstprivate(rhos_buffer) reduction(vec_plus:current_density_time) schedule(dynamic)
        for (int i = 0; i < n_z; ++i) {
            PROGRESS_BAR_UPDATE(n_z);

            momentum_type k = momentum_type::SymmetrizedRandom(gens[omp_get_thread_num()]);
            if (k.is_dirac_point()) {
                continue;
            }

            __time_evolution__(rhos_buffer, laser, k, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                current_density_time[i] += rhos_buffer[i] * std::sin(k.z - laser->laser_function(i * time_step));
            }

            k.invert();
            __time_evolution__(rhos_buffer, laser, k, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                current_density_time[i] += rhos_buffer[i] * std::sin(k.z - laser->laser_function(i * time_step));
            }
        }
        for (auto& j : current_density_time) {
            j *= (pi*pi*pi)/n_z;
        }
#endif
        return current_density_time;
    }
}