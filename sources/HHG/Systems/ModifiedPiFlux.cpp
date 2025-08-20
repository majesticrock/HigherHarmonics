#include "ModifiedPiFlux.hpp"
#include "../GeneralMagnus.hpp"
#include "../Laser/gauss.hpp"
#include "../thread_gauss.hpp"

#include <cmath>
#include <cassert>
#include <numeric>
#include <random>
#include <omp.h>
#include <functional> 

#include <mrock/utility/progress_bar.hpp>

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;

typedef runge_kutta_fehlberg78<HHG::Systems::ModifiedPiFlux::sigma_state_type> sigma_error_stepper_type;

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
constexpr double RESCUE_TRAFO = 1e-4;


namespace HHG::Systems {
    ModifiedPiFlux::ModifiedPiFlux(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _diagonal_relaxation_time, h_float _offdiagonal_relaxation_time)
        : beta(is_zero(temperature) ? std::numeric_limits<h_float>::max() : _photon_energy / (k_B * temperature)), 
            E_F(_E_F / _photon_energy), 
            hopping_element(_band_width / sqrt_12), 
            lattice_constant(sqrt_3 * hbar * _v_F / (_photon_energy * _band_width)),
            inverse_diagonal_relaxation_time((1e15 * hbar) / (_diagonal_relaxation_time * _photon_energy)),
            inverse_offdiagonal_relaxation_time((1e15 * hbar) / (_offdiagonal_relaxation_time * _photon_energy))
    {
        //gauss::precompute<100>();
        //abort();
    }

    void ModifiedPiFlux::time_evolution_sigma(nd_vector &rhos, Laser::Laser const *const laser, const momentum_type &k, const TimeIntegrationConfig &time_config) const
    {
        const h_float prefactor = 4 * hopping_element;

        const h_float alpha2 = occupation_a(k);
        const h_float beta2 = occupation_b(k);
        const h_float alpha_beta_diff = alpha2 - beta2;
        const h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);
        const h_float z_epsilon = k.C_z + dispersion(k);

        sigma_state_type current_state = { ic_sigma_x(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_y(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_z(k, alpha_beta_diff, alpha_beta_prod, z_epsilon) };

        auto right_side = [&k, &laser, &prefactor](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const sigma_state_type m = {k.C_x, k.C_y, std::cos(k.z - laser->laser_function(t))};
            dxdt = prefactor * m.cross(state);
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

    void ModifiedPiFlux::time_evolution_diagonal_relaxation(nd_vector &rhos, Laser::Laser const * const laser, const momentum_type &k, const TimeIntegrationConfig &time_config) const
    {
        assert(!is_zero(dispersion(k)));
        const h_float prefactor = 4 * hopping_element;
        
        h_float alpha2 = occupation_a(k);
        h_float beta2 = occupation_b(k);
        h_float alpha_beta_diff = alpha2 - beta2;
        h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);
        h_float alpha_beta_imag{};
        h_float z_epsilon = k.C_z + dispersion(k);
        h_float normalization = dispersion(k) * z_epsilon;
        
        sigma_state_type relax_to_diagonal;
        sigma_state_type relax_to_offdiagonal;

        sigma_state_type current_state = ic_sigma(k, alpha_beta_diff, alpha_beta_prod, z_epsilon);

        auto update_equilibrium_state = [&](const h_float laser_at_t) {
            auto shifted_k = k;
            shifted_k.update_z(k.z - laser_at_t);
            if (is_zero(shifted_k.C_x) && is_zero(shifted_k.C_y) && shifted_k.C_z < h_float{}) {
                shifted_k.C_x = std::abs(shifted_k.C_z) * RESCUE_TRAFO;
                shifted_k.C_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            
            alpha2 = occupation_a(shifted_k);
            beta2 = occupation_b(shifted_k);
            alpha_beta_diff = alpha2 - beta2;

            const std::array<h_float, 3> sigmas = diagonal_sigma(current_state, shifted_k);
            const h_float xy_length = sqrt(sigmas[0]*sigmas[0] + sigmas[1]*sigmas[1]);
            alpha_beta_prod = 2 * sqrt(alpha2 * beta2) * ( is_zero(xy_length) ? 1.0 : sigmas[0] / xy_length );
            alpha_beta_imag = 2 * sqrt(alpha2 * beta2) * ( is_zero(xy_length) ? 0.0 : sigmas[1] / xy_length );

            z_epsilon = shifted_k.C_z + dispersion(shifted_k);
            normalization = dispersion(shifted_k) * z_epsilon;
        
            relax_to_diagonal(0) = shifted_k.C_x;
            relax_to_diagonal(1) = shifted_k.C_y;
            relax_to_diagonal(2) = shifted_k.C_z;
            relax_to_diagonal *= (alpha_beta_diff * z_epsilon / normalization);
        
            relax_to_offdiagonal(0) = alpha_beta_prod * (  shifted_k.C_y * shifted_k.C_y + shifted_k.C_z * z_epsilon);
            relax_to_offdiagonal(1) = alpha_beta_prod * (- shifted_k.C_x * shifted_k.C_y);
            relax_to_offdiagonal(2) = alpha_beta_prod * (- shifted_k.C_x * z_epsilon);

            relax_to_offdiagonal(0) += alpha_beta_imag * (- shifted_k.C_x * shifted_k.C_y); 
            relax_to_offdiagonal(1) += alpha_beta_imag * (  shifted_k.C_x * shifted_k.C_x + shifted_k.C_z * z_epsilon);
            relax_to_offdiagonal(2) += alpha_beta_imag * (- shifted_k.C_y * z_epsilon);
            relax_to_offdiagonal /= normalization;
        };

        update_equilibrium_state(laser->laser_function(time_config.t_begin));

        auto right_side = [this, &k, &laser, &prefactor, &update_equilibrium_state, &relax_to_diagonal, &relax_to_offdiagonal](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const sigma_state_type m = {k.C_x, k.C_y, std::cos(k.z - laser->laser_function(t))};
            update_equilibrium_state(laser->laser_function(t));
            dxdt = prefactor * m.cross(state) 
                - inverse_diagonal_relaxation_time * (state / 3.0 - relax_to_diagonal) // sigma^z
                - inverse_offdiagonal_relaxation_time * (state * (2. / 3.) - relax_to_offdiagonal); // sigma^x and sigma^y -> therefore 2*state
        };

        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.resize(time_config.n_measurements + 1);
        rhos[0] = current_state(2);

        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            rhos[i] = current_state(2);
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    std::vector<h_float> ModifiedPiFlux::compute_current_density(Laser::Laser const *const laser, TimeIntegrationConfig const &time_config, const int rank, const int n_ranks, const int n_z) const
    {
        return current_density_continuum_limit(laser, time_config, rank, n_ranks, n_z);
    }

    std::string ModifiedPiFlux::info() const
    {
        return  "ModifiedPiFlux\nT=" + std::to_string(1.0 / beta) + "\n" 
                + "E_F=" + std::to_string(E_F) + "\n" 
                + "t=" + std::to_string(hopping_element) + "\n" 
                + "d=" + std::to_string(lattice_constant) + "\n";
    }

    h_float ModifiedPiFlux::dispersion(const momentum_type& k) const
    {
        return sqrt(k.C_x*k.C_x + k.C_y*k.C_y + k.C_z*k.C_z);
    }

    std::string ModifiedPiFlux::get_property_in_SI_units(const std::string& property, const h_float photon_energy) const
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

    h_float ModifiedPiFlux::occupation_a(const momentum_type& k) const 
    {
        return fermi_function(-E_F + 2 * hopping_element * dispersion(k), beta);
    }

    h_float ModifiedPiFlux::occupation_b(const momentum_type& k) const
    {
        return fermi_function(-E_F - 2 * hopping_element * dispersion(k), beta);
    }

    ModifiedPiFlux::momentum_type::momentum_type(h_float x, h_float y, h_float z) noexcept
        : C_x(1. - std::cos(x)), C_y(1. - std::cos(y)), C_z(1. - std::cos(z)), z(z)
    { }

    void ModifiedPiFlux::momentum_type::update(h_float x, h_float y, h_float z) noexcept
    {
        this->C_x = 1. - std::cos(x);
        this->C_y = 1. - std::cos(y);
        this->C_z = 1. - std::cos(z);
        this->z = z;
    }

    void ModifiedPiFlux::momentum_type::update_x(h_float val) noexcept
    {
        this->C_x = 1. - std::cos(val);
    }

    void ModifiedPiFlux::momentum_type::update_y(h_float val) noexcept
    {
        this->C_y = 1. - std::cos(val);
    }

    void ModifiedPiFlux::momentum_type::update_z(h_float val) noexcept
    {
        this->C_z = 1. - std::cos(val);
        this->z = val;
    }

    void ModifiedPiFlux::__time_evolution__(nd_vector& rhos, Laser::Laser const * const laser, 
        const momentum_type& k, const TimeIntegrationConfig& time_config) const
    {
        if (inverse_diagonal_relaxation_time > h_float{}) {
            return time_evolution_diagonal_relaxation(rhos, laser, k, time_config);
        }
        return time_evolution_sigma(rhos, laser, k, time_config);
    }

    h_float ModifiedPiFlux::ic_sigma_x(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept
    {
        return (alpha_beta_diff * k.C_x * z_epsilon + alpha_beta_prod * (k.C_y * k.C_y + k.C_z * z_epsilon)) / (dispersion(k) * z_epsilon);
    }

    h_float ModifiedPiFlux::ic_sigma_y(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept
    {
        return (alpha_beta_diff * k.C_y * z_epsilon - alpha_beta_prod * k.C_x * k.C_y) / (dispersion(k) * z_epsilon);
    }

    h_float ModifiedPiFlux::ic_sigma_z(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept
    {
        return (alpha_beta_diff * k.C_z * z_epsilon - alpha_beta_prod * k.C_x * z_epsilon) / (dispersion(k) * z_epsilon);
    }

    ModifiedPiFlux::sigma_state_type ModifiedPiFlux::ic_sigma(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept
    {
        return sigma_state_type{ic_sigma_x(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_y(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_z(k, alpha_beta_diff, alpha_beta_prod, z_epsilon)};
    }

    nd_vector ModifiedPiFlux::improved_xy_integral(momentum_type& k, nd_vector& rhos_buffer, Laser::Laser const * const laser, TimeIntegrationConfig const& time_config) const {
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

    std::vector<h_float> ModifiedPiFlux::current_density_continuum_limit(Laser::Laser const * const laser, TimeIntegrationConfig const& time_config, 
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
            if (is_zero(k.C_x) && is_zero(k.C_y) && k.C_z < h_float{}) {
                k.C_x = std::abs(k.C_z) * RESCUE_TRAFO;
                k.C_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }

            x_buffer = improved_xy_integral(k, rhos_buffer, laser, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                x_buffer[i] *= z_gauss::weights[z] * std::sin(k.z - laser->laser_function(i * time_step + time_config.t_begin));
            }
            std::transform(current_density_time.begin(), current_density_time.end(), x_buffer.begin(), current_density_time.begin(), std::plus<>());
        
            /*
            *  -z
            */
            k.update_z(-pi * z_gauss::abscissa[z]);
            if (is_zero(k.C_x) && is_zero(k.C_y) && k.C_z < h_float{}) {
                k.C_x = std::abs(k.C_z) * RESCUE_TRAFO;
                k.C_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }

            x_buffer = improved_xy_integral(k, rhos_buffer, laser, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                x_buffer[i] *= z_gauss::weights[z] * std::sin(k.z - laser->laser_function(i * time_step + time_config.t_begin));
            }
            std::transform(current_density_time.begin(), current_density_time.end(), x_buffer.begin(), current_density_time.begin(), std::plus<>());
        }
        return current_density_time;
    }

    std::vector<OccupationContainer> ModifiedPiFlux::compute_occupation_numbers(Laser::Laser const * const laser, 
        TimeIntegrationConfig const& time_config, const int N) const
    {
        const h_float dx = 2. * pi / N;
        const h_float dz = 2. * pi / N;
        
        auto k_xy = [dx](int x) {
            return x * dx - pi;
        };
        auto k_z = [dz](int z) {
            return dz * z - pi;
        };

        auto occupations = [this](sigma_state_type const& input, momentum_type const& k) {
            const auto diags = diagonal_sigma(input, k);

            return OccupationContainer::occupation_t{ 
                std::sqrt( 
                    0.5 * ( -diags[2] + norm(diags[0], diags[1], diags[2] ) )
                ),
                std::sqrt( 
                    0.5 * ( diags[2] + norm(diags[0], diags[1], diags[2] ) )
                )
            };
        };

        auto coordinate_shift = [N, dz](int z, h_float laser_value) -> int {
            const int shift = static_cast<int>(std::round(laser_value / dz));
            return (((z - shift) % N) + N) % N;
        };

        const h_float prefactor = 4. * hopping_element;
        std::vector<OccupationContainer> computed_occupations(time_config.n_measurements + 1, OccupationContainer(N));

        std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for
        for (int i = 0; i < N * N; ++i) {
            PROGRESS_BAR_UPDATE(N*N);

            const int x = i / N;
            const int z = i % N;

            momentum_type k(k_xy(x), 0.0, k_z(z));
            if (is_zero(k.C_x) && is_zero(k.C_y) && k.C_z < h_float{}) {
                k.C_x = std::abs(k.C_z) * RESCUE_TRAFO;
                k.C_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            momentum_type shifted_k = k;

            h_float alpha2 = occupation_a(k);
            h_float beta2 = occupation_b(k);
            h_float alpha_beta_diff = alpha2 - beta2;
            h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);
            h_float alpha_beta_imag{};
            h_float z_epsilon = k.C_z + dispersion(k);
            h_float normalization = dispersion(k) * z_epsilon;
            
            sigma_state_type relax_to_diagonal;
            sigma_state_type relax_to_offdiagonal;
            
            sigma_state_type current_state = ic_sigma(k, alpha_beta_diff, alpha_beta_prod, z_epsilon);
            
            auto update_equilibrium_state = [&](const h_float laser_at_t, const h_float __t) {
                shifted_k.update_z(k.z - laser_at_t);

                if (is_zero(shifted_k.C_x) && is_zero(shifted_k.C_y) && shifted_k.C_z < h_float{}) {
                    shifted_k.C_x = std::abs(shifted_k.C_z) * RESCUE_TRAFO;
                    shifted_k.C_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
                }
                
                alpha2 = occupation_a(shifted_k);
                beta2 = occupation_b(shifted_k);
                alpha_beta_diff = alpha2 - beta2;

                const std::array<h_float, 3> sigmas = diagonal_sigma(current_state, shifted_k);
                const h_float xy_length = sqrt(sigmas[0]*sigmas[0] + sigmas[1]*sigmas[1]);
                alpha_beta_prod = 2 * sqrt(alpha2 * beta2) * ( is_zero(xy_length) ? 1.0 : sigmas[0] / xy_length );
                alpha_beta_imag = 2 * sqrt(alpha2 * beta2) * ( is_zero(xy_length) ? 0.0 : sigmas[1] / xy_length );

                z_epsilon = shifted_k.C_z + dispersion(shifted_k);
                normalization = dispersion(shifted_k) * z_epsilon;
            
                relax_to_diagonal(0) = shifted_k.C_x;
                relax_to_diagonal(1) = shifted_k.C_y;
                relax_to_diagonal(2) = shifted_k.C_z;
                relax_to_diagonal *= (alpha_beta_diff * z_epsilon / normalization);
            
                relax_to_offdiagonal(0) = alpha_beta_prod * (  shifted_k.C_y * shifted_k.C_y + shifted_k.C_z * z_epsilon);
                relax_to_offdiagonal(1) = alpha_beta_prod * (- shifted_k.C_x * shifted_k.C_y);
                relax_to_offdiagonal(2) = alpha_beta_prod * (- shifted_k.C_x * z_epsilon);

                relax_to_offdiagonal(0) += alpha_beta_imag * (- shifted_k.C_x * shifted_k.C_y); 
                relax_to_offdiagonal(1) += alpha_beta_imag * (  shifted_k.C_x * shifted_k.C_x + shifted_k.C_z * z_epsilon);
                relax_to_offdiagonal(2) += alpha_beta_imag * (- shifted_k.C_y * z_epsilon);
                relax_to_offdiagonal /= normalization;
            };

            update_equilibrium_state(laser->laser_function(time_config.t_begin), 0.0);

            auto right_side = [this, &k, &laser, &prefactor, &update_equilibrium_state, &relax_to_diagonal, &relax_to_offdiagonal](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
                const sigma_state_type m = {k.C_x, k.C_y, std::cos(k.z - laser->laser_function(t))};
                update_equilibrium_state(laser->laser_function(t), t);

                dxdt = prefactor * m.cross(state) 
                    - inverse_diagonal_relaxation_time * (state / 3.0 - relax_to_diagonal) // sigma^z
                    - inverse_offdiagonal_relaxation_time * (state * (2. / 3.) - relax_to_offdiagonal); // sigma^x and sigma^y -> therefore 2*state
            };
        
            const h_float measure_every = time_config.measure_every();
            const h_float dt = time_config.dt();
            h_float t_begin = time_config.t_begin;
            h_float t_end = t_begin + measure_every;
        
            computed_occupations[0](x, coordinate_shift(z, laser->laser_function(t_begin))) = occupations(current_state, shifted_k);

            runge_kutta4< sigma_state_type > stepper;

            for (int i = 1; i <= time_config.n_measurements; ++i) {
                integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
                //integrate_const(stepper, right_side, current_state, t_begin, t_end, dt);
                computed_occupations[i](x, coordinate_shift(z, laser->laser_function(t_begin))) = occupations(current_state, shifted_k);

                t_begin = t_end;
                t_end += measure_every;
            }
        }
        return computed_occupations;
    }

    std::pair<std::vector<h_float>, std::vector<h_float>> ModifiedPiFlux::current_per_energy(Laser::Laser const * const laser, 
            TimeIntegrationConfig const& time_config, const int N) const
    {
        constexpr h_float energy_cut = 0.5 * sqrt_3;

        const h_float dxy = 2 * pi / N;
        const h_float dz  = 2 * pi / N;
        
        auto k_xy = [dxy](int x) {
            return x * dxy - pi;
        };
        auto k_z = [dz](int z) {
            return dz * z - pi;
        };

        const h_float time_step = time_config.measure_every();

        nd_vector rhos_buffer;
        std::vector<h_float> current_density_low(time_config.n_measurements + 1, h_float{});
        std::vector<h_float> current_density_high(time_config.n_measurements + 1, h_float{});

        std::vector<int> progresses(omp_get_max_threads(), int{});
#pragma omp parallel for private(rhos_buffer) reduction(vec_plus:current_density_low,current_density_high)
        for (int i = 0; i < N * N * N; ++i) {
            PROGRESS_BAR_UPDATE(N*N*N);

            const int z = i / (N * N);
            const int y = (i / N) % N;
            const int x = i % N;

            momentum_type k(k_xy(x), k_xy(y), k_z(z));
            if (is_zero(k.C_x) && is_zero(k.C_y) && k.C_z < h_float{}) {
                k.C_x = std::abs(k.C_z) * RESCUE_TRAFO;
                k.C_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            
            if (is_zero(k.C_x) && is_zero(k.C_y) && k.C_z < h_float{}) {
                k.C_x = std::abs(k.C_z) * RESCUE_TRAFO;
                k.C_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            __time_evolution__(rhos_buffer, laser, k, time_config);

            for (int i = 0; i <= time_config.n_measurements; ++i) {
                const h_float laser_value = laser->laser_function(i * time_step + time_config.t_begin);
                const h_float instantaneous_energy = dispersion(k.shift_z(laser_value));
                if (instantaneous_energy < energy_cut) {
                    current_density_low[i] = rhos_buffer(i) * std::sin(k.z - laser_value);
                }
                else {
                    current_density_high[i] = rhos_buffer(i) * std::sin(k.z - laser_value);
                }
            }
            

            /*
            *  -z
            */
            k.invert();
            if (is_zero(k.C_x) && is_zero(k.C_y) && k.C_z < h_float{}) {
                k.C_x = std::abs(k.C_z) * RESCUE_TRAFO;
                k.C_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            __time_evolution__(rhos_buffer, laser, k, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                const h_float laser_value = laser->laser_function(i * time_step + time_config.t_begin);
                const h_float instantaneous_energy = dispersion(k.shift_z(laser_value));
                if (instantaneous_energy < energy_cut) {
                    current_density_low[i] = rhos_buffer(i) * std::sin(k.z - laser_value);
                }
                else {
                    current_density_high[i] = rhos_buffer(i) * std::sin(k.z - laser_value);
                }
            }
        }

        for (auto& val : current_density_low) {
            val /= N*N*N;
        }
        for (auto& val : current_density_high) {
            val /= N*N*N;
        }
        return {current_density_low, current_density_high};
    }

    std::array<h_float, 3> ModifiedPiFlux::diagonal_sigma(sigma_state_type const& input, momentum_type const& k) const
    {
        const h_float z = k.C_z + dispersion(k);
        const h_float norm = 2 * dispersion(k) * z;
        const h_float __xz = 2 * k.C_x * z;
        const h_float __yz = 2 * k.C_y * z;
        const h_float __xy = 2 * k.C_x * k.C_y;

        return { 
            (input(0) * (z*z - k.C_x*k.C_x + k.C_y*k.C_y) - __xy * input(1) - __xz * input(2)) / norm,
            (input(1) * (z*z + k.C_x*k.C_x - k.C_y*k.C_y) - __xy * input(0) - __yz * input(2)) / norm,
            (input(2) * (z*z - k.C_x*k.C_x - k.C_y*k.C_y) + __xz * input(0) + __yz * input(1)) / norm 
        };
    }
}