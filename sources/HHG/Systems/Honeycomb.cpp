#include "Honeycomb.hpp"
#include "../Laser/gauss.hpp"
#include "../thread_gauss.hpp"

#include <cassert>

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
#define PROGRESS_BAR_UPDATE(k_max) ++(progresses[omp_get_thread_num()]); \
            if (omp_get_thread_num() == 0) { \
                mrock::utility::progress_bar( \
                    static_cast<float>(std::reduce(progresses.begin(), progresses.end())) / static_cast<float>((k_max)) \
                ); \
            }
#else
#define PROGRESS_BAR_UPDATE(k_max)
#endif

namespace HHG::Systems {
    Honeycomb::momentum_type::momentum_type(h_float x, h_float y) noexcept
        : x(x), y(y) 
    {
        this->set_gamma();
    }

    void Honeycomb::momentum_type::set_gamma() noexcept
    {
        this->gamma = h_complex{};
        for(const auto& nn : Honeycomb::nearest_neighbors) {
            this->gamma += std::exp(imaginary_unit * (nn[0] * this->x + nn[1] * this->y));
        }
    }

    h_complex Honeycomb::momentum_type::shifted_gamma(h_float x_shift, h_float y_shift) const noexcept
    {
        h_complex gamma_shift{};
        for(const auto& nn : Honeycomb::nearest_neighbors) {
            gamma_shift += std::exp(imaginary_unit * (nn[0] * (this->x - x_shift) + nn[1] * (this->y - y_shift)));
        }
        return gamma_shift;
    }

    std::array<h_complex, 2> Honeycomb::momentum_type::shifted_grad_gamma(h_float x_shift, h_float y_shift) const noexcept
    {
        std::array<h_complex, 2> gamma_shift{};
        h_complex buffer{};
        for(const auto& nn : Honeycomb::nearest_neighbors) {
            buffer = std::exp(imaginary_unit * (nn[0] * (this->x - x_shift) + nn[1] * (this->y - y_shift)));
            gamma_shift[0] += nn[0] * buffer;
            gamma_shift[1] += nn[1] * buffer;
        }
        return gamma_shift;
    }

    void Honeycomb::momentum_type::update(h_float x, h_float y) noexcept
    {
        this->x = x;
        this->y = y;
        this->set_gamma();
    }

    void Honeycomb::momentum_type::update_x(h_float val) noexcept
    {
        this->x = val;
        this->set_gamma();
    }

    void Honeycomb::momentum_type::update_y(h_float val) noexcept
    {
        this->y = val;
        this->set_gamma();
    }

    Honeycomb::momentum_type Honeycomb::momentum_type::shift(h_float x_shift, h_float y_shift) const noexcept
    {
        return momentum_type(this->x - x_shift, this->y - y_shift);
    }

    Honeycomb::Honeycomb(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _decay_time)
        : beta(is_zero(temperature) ? std::numeric_limits<h_float>::infinity() : _photon_energy / (k_B * temperature)), 
            E_F(_E_F / _photon_energy), 
            hopping_element(_band_width / 6), 
            lattice_constant(4 * hbar * _v_F / (_photon_energy * _band_width)),
            inverse_decay_time((1e15 * hbar) / (_decay_time * _photon_energy))
    {}

    std::string Honeycomb::get_property_in_SI_units(const std::string &property, const h_float photon_energy) const
    {
        if (property == "E_F") {
            return std::to_string(E_F * photon_energy) + " meV";
        }
        else if (property == "t") {
            return std::to_string(hopping_element * photon_energy) + " meV";
        }
        else if (property == "d") {
            return std::to_string(1e12 * lattice_constant) + " pm";
        }
        else if (property == "beta") {
            return std::to_string(beta / photon_energy) + " meV^-1";
        }
        else if (property == "T") {
            return std::to_string(photon_energy / (k_B * beta)) + " K";
        }
        else {
            throw std::invalid_argument("Property '" + property + "' is not recognized!");
        }
    }

    std::array<std::vector<h_float>, n_debug_points> Honeycomb::compute_current_density_debug(Laser::Laser const * const laser, 
            TimeIntegrationConfig const& time_config, const int n_z) const
    {
        throw std::runtime_error("Honeycomb::compute_current_density_debug is not implemented yet!");
    }

    void Honeycomb::time_evolution_sigma(nd_vector& rho_x, nd_vector& rho_y, Laser::Laser const * const laser, 
        const momentum_type& k, const TimeIntegrationConfig& time_config) const
    {
        const h_float prefactor = 2 * hopping_element;

        const h_float alpha2 = occupation_a(k);
        const h_float beta2 = occupation_b(k);
        const h_float alpha_beta_diff = alpha2 - beta2;
        const h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);

        sigma_state_type current_state = ic_sigma(k, alpha_beta_diff, alpha_beta_prod);

        auto right_side = [this, &k, &laser, &prefactor](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const h_complex buffer = k.shifted_gamma(cos_theta * laser->laser_function(t), sin_theta * laser->laser_function(t));
            dxdt(0) = -prefactor * (state[2] * buffer.imag());
            dxdt(1) = -prefactor * (state[2] * buffer.real());
            dxdt(2) = prefactor * (state[0] * buffer.imag() + state[1] * buffer.real());
        };

        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rho_x.conservativeResize(time_config.n_measurements + 1);
        rho_x[0] = current_state(0);
        rho_y.conservativeResize(time_config.n_measurements + 1);
        rho_y[0] = current_state(1);

        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            rho_x[i] = current_state(0);
            rho_y[i] = current_state(1);
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void Honeycomb::time_evolution_decay(nd_vector& rho_x, nd_vector& rho_y, Laser::Laser const * const laser, 
        const momentum_type& k, const TimeIntegrationConfig& time_config) const
    {
        const h_float prefactor = 2 * hopping_element;
        sigma_state_type equilibrium_state;

        auto update_equilibrium_state = [this, &k, &equilibrium_state](const h_float laser_at_t) {
            // Something is not quite right - a constant (finite) vector potential currently causes a constant current density.
            const auto shifted_k = k.shift(cos_theta * laser_at_t, sin_theta * laser_at_t);
            const h_float alpha2 = occupation_a(shifted_k);
            const h_float beta2 = occupation_b(shifted_k);
            const h_float alpha_beta_diff = alpha2 - beta2;
            const h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);
            equilibrium_state = ic_sigma(shifted_k, alpha_beta_diff, alpha_beta_prod);
        };

        auto right_side = [this, &k, &laser, &prefactor, &equilibrium_state, &update_equilibrium_state](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const h_float laser_value = laser->laser_function(t);
            update_equilibrium_state(laser_value);
            const h_complex buffer = k.shifted_gamma(cos_theta * laser_value, sin_theta * laser_value);
            dxdt(0) = -prefactor * (state[2] * buffer.imag());
            dxdt(1) = -prefactor * (state[2] * buffer.real());
            dxdt(2) = prefactor * (state[0] * buffer.imag() + state[1] * buffer.real());

            dxdt -= inverse_decay_time * (state - equilibrium_state);
        };

        update_equilibrium_state(laser->laser_function(time_config.t_begin));
        sigma_state_type current_state = equilibrium_state;

        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rho_x.conservativeResize(time_config.n_measurements + 1);
        rho_x[0] = current_state(0);
        rho_y.conservativeResize(time_config.n_measurements + 1);
        rho_y[0] = current_state(1);

        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            rho_x[i] = current_state(0);
            rho_y[i] = current_state(1);
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void Honeycomb::__time_evolution__(nd_vector& rho_x, nd_vector& rho_y, Laser::Laser const * const laser, 
        const momentum_type& k, const TimeIntegrationConfig& time_config) const
    {
        if (inverse_decay_time > h_float{}) {
            return time_evolution_decay(rho_x, rho_y, laser, k, time_config);
        }
        return time_evolution_sigma(rho_x, rho_y, laser, k, time_config);
    }

    std::vector<h_float> Honeycomb::compute_current_density(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
        const int rank, const int n_ranks, const int n_x) const
    {
        constexpr h_float min_x = - 2 * pi / 3;
        constexpr h_float max_x = 2 * pi / 3;
        auto bound_y = [](h_float x) {
            return (4 * pi / (3 * sqrt_3)) - (std::abs(x) / sqrt_3);
        };

        auto transform = [](h_float x, h_float low, h_float high) {
            return 0.5 * (high - low) * x + 0.5 * (high + low);
        };
        auto weight = [](h_float low, h_float high) {
            return 0.5 * (high - low);
        };

        using __gauss = gauss::container<2 * N_k>;

        nd_vector rho_x;
        nd_vector rho_y;
        std::vector<h_float> buffer_x(time_config.n_measurements + 1, h_float{});
        std::vector<h_float> buffer_y(time_config.n_measurements + 1, h_float{});
        std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});

        const h_float time_step = time_config.measure_every();
#ifdef NO_MPI
        std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for private(rho_x, rho_y) reduction(vec_plus:buffer_x) reduction(vec_plus:buffer_y)
        for (int x = 0; x < N_k; ++x)
#else
        int jobs_per_rank = N_k / n_ranks;
        if (rank == 0) { std::cout << "Jobs per rank: " << jobs_per_rank << "\nn_ranks: " << n_ranks << "\nN_k: " << N_k << std::endl; }
        if (jobs_per_rank * n_ranks < N_k) ++jobs_per_rank;
        const int this_rank_min_x = rank * jobs_per_rank;
        const int this_rank_max_x = this_rank_min_x + jobs_per_rank > N_k ? N_k : this_rank_min_x + jobs_per_rank;
        for (int x = this_rank_min_x; x < this_rank_max_x; ++x)
#endif
        {
            PROGRESS_BAR_UPDATE(N_k);

            momentum_type k(transform(__gauss::abscissa[x], min_x, max_x), h_float{});
            const h_float curr_max_y = bound_y(k.x);

            for (int y = 0; y < N_k; ++y) {
                const h_float transform_weight = weight(min_x, max_x) * weight(-curr_max_y, curr_max_y) * __gauss::weights[x] * __gauss::weights[y];

                k.update_y(transform(__gauss::abscissa[y], -curr_max_y, curr_max_y));
                __time_evolution__(rho_x, rho_y, laser, k, time_config);
                
                for (int i = 0; i <= time_config.n_measurements; ++i) {
                    const auto grad_gamma = k.shifted_grad_gamma(cos_theta * laser->laser_function(time_config.t_begin + i * time_step), sin_theta * laser->laser_function(time_config.t_begin + i * time_step));
                    buffer_x[i] += transform_weight * (grad_gamma[0].real() * rho_x[i] + grad_gamma[0].imag() * rho_y[i]);
                    buffer_y[i] += transform_weight * (grad_gamma[1].real() * rho_x[i] + grad_gamma[1].imag() * rho_y[i]);
                }

                // minus y
                k.update_y(transform(-__gauss::abscissa[y], -curr_max_y, curr_max_y));
                __time_evolution__(rho_x, rho_y, laser, k, time_config);

                for (int i = 0; i <= time_config.n_measurements; ++i) {
                    const auto grad_gamma = k.shifted_grad_gamma(cos_theta * laser->laser_function(time_config.t_begin + i * time_step), sin_theta * laser->laser_function(time_config.t_begin + i * time_step));
                    buffer_x[i] += transform_weight * (grad_gamma[0].real() * rho_x[i] + grad_gamma[0].imag() * rho_y[i]);
                    buffer_y[i] += transform_weight * (grad_gamma[1].real() * rho_x[i] + grad_gamma[1].imag() * rho_y[i]);
                }
            }

            // minus x
            k.update_x(transform(-__gauss::abscissa[x], min_x, max_x));
            for (int y = 0; y < N_k; ++y) {
                const h_float transform_weight = weight(min_x, max_x) * weight(-curr_max_y, curr_max_y) * __gauss::weights[x] * __gauss::weights[y];

                k.update_y(transform(__gauss::abscissa[y], -curr_max_y, curr_max_y));
                __time_evolution__(rho_x, rho_y, laser, k, time_config);

                for (int i = 0; i <= time_config.n_measurements; ++i) {
                    const auto grad_gamma = k.shifted_grad_gamma(cos_theta * laser->laser_function(time_config.t_begin + i * time_step), sin_theta * laser->laser_function(time_config.t_begin + i * time_step));
                    buffer_x[i] += transform_weight * (grad_gamma[0].real() * rho_x[i] + grad_gamma[0].imag() * rho_y[i]);
                    buffer_y[i] += transform_weight * (grad_gamma[1].real() * rho_x[i] + grad_gamma[1].imag() * rho_y[i]);
                }

                // minus y
                k.update_y(transform(-__gauss::abscissa[y], -curr_max_y, curr_max_y));
                __time_evolution__(rho_x, rho_y, laser, k, time_config);
                          
                for (int i = 0; i <= time_config.n_measurements; ++i) {
                    const auto grad_gamma = k.shifted_grad_gamma(cos_theta * laser->laser_function(time_config.t_begin + i * time_step), sin_theta * laser->laser_function(time_config.t_begin + i * time_step));
                    buffer_x[i] += transform_weight * (grad_gamma[0].real() * rho_x[i] + grad_gamma[0].imag() * rho_y[i]);
                    buffer_y[i] += transform_weight * (grad_gamma[1].real() * rho_x[i] + grad_gamma[1].imag() * rho_y[i]);
                }
            }
        }

        // Project the current density onto the laser polarization
        // I assume that the current density will vanish orthogonal to the laser polarization, but this is yet to be verified
        // 28.05.2025: Seems to be correct
        for (int i = 0; i <= time_config.n_measurements; ++i) {
            current_density_time[i] = buffer_x[i] * cos_theta + buffer_y[i] * sin_theta;
            const auto j_perp = buffer_x[i] * sin_theta + buffer_y[i] * cos_theta;
            if(std::abs(j_perp) > 1e-6 * std::abs(current_density_time[i]) && std::abs(j_perp) > 1e-6) {
                std::cerr << "Warning: j_perp = " << j_perp << "\t\tj_par = " << current_density_time[i] << "!\n";
            }
        }
        return current_density_time;
    }
    
    std::string Honeycomb::info() const
    {
        return  "Honeycomb\nT=" + std::to_string(1.0 / beta) + "\n" 
                + "E_F=" + std::to_string(E_F) + "\n" 
                + "t=" + std::to_string(hopping_element) + "\n" 
                + "d=" + std::to_string(lattice_constant) + "\n";
    }

    h_float Honeycomb::dispersion(const momentum_type& k) const
    {
        return std::abs(k.gamma);
    }

    h_float Honeycomb::occupation_a(const momentum_type& k) const 
    {
        return fermi_function(E_F - hopping_element * dispersion(k), beta);
    }

    h_float Honeycomb::occupation_b(const momentum_type& k) const
    {
        return fermi_function(E_F + hopping_element * dispersion(k), beta);
    }


    Eigen::Vector<HHG::h_float, 3> Honeycomb::ic_sigma(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod) const noexcept
    {
        assert(!k.is_dirac_point());
        return Eigen::Vector<HHG::h_float, 3>{ 
            -alpha_beta_diff * k.gamma.real() / std::abs(k.gamma), 
            alpha_beta_diff * k.gamma.imag() / std::abs(k.gamma), 
            -alpha_beta_prod };
    }
}