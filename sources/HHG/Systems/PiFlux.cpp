#include "PiFlux.hpp"
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

typedef runge_kutta_fehlberg78<HHG::Systems::PiFlux::sigma_state_type> sigma_error_stepper_type;

constexpr HHG::h_float abs_error = 1.0e-12;
constexpr HHG::h_float rel_error = 1.0e-8;

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size(), decltype(omp_orig)::value_type{}))

#if defined(NO_MPI) && !defined(MROCK_CL1_CASCADE)
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

#define DDT_J

namespace HHG::Systems {
    PiFlux::PiFlux(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy, h_float _diagonal_relaxation_time, h_float _offdiagonal_relaxation_time)
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

        auto right_side = [&k, &laser, &prefactor](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const sigma_state_type m = {k.cos_x, k.cos_y, std::cos(k.z - laser->laser_function(t))};
            dxdt = cross_product(m, state);
            for (auto& d : dxdt) d *= prefactor;
        };

        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.conservativeResize(time_config.n_measurements + 1);
        rhos[0] = current_state[2];

        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            rhos[i] = current_state[2];
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void PiFlux::time_evolution_diagonal_relaxation(nd_vector &rhos, Laser::Laser const * const laser, const momentum_type &k, const TimeIntegrationConfig &time_config) const
    {
        assert(!is_zero(dispersion(k)));
        const h_float prefactor = 4 * hopping_element;
        
        h_float alpha2 = occupation_a(k);
        h_float beta2 = occupation_b(k);
        h_float alpha_beta_diff = alpha2 - beta2;
        h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);
        h_float alpha_beta_imag{};
        h_float z_epsilon = k.cos_z + dispersion(k);
        h_float normalization = dispersion(k) * z_epsilon;
        
        sigma_state_type relax_to_diagonal;
        sigma_state_type relax_to_offdiagonal;

        sigma_state_type current_state = ic_sigma(k, alpha_beta_diff, alpha_beta_prod, z_epsilon);

        auto update_equilibrium_state = [&](const h_float laser_at_t) {
            auto shifted_k = k;
            shifted_k.update_z(k.z - laser_at_t);
            if (is_zero(shifted_k.cos_x) && is_zero(shifted_k.cos_y) && shifted_k.cos_z < h_float{}) {
                shifted_k.cos_x = std::abs(shifted_k.cos_z) * RESCUE_TRAFO;
                shifted_k.cos_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            
            alpha2 = occupation_a(shifted_k);
            beta2 = occupation_b(shifted_k);
            alpha_beta_diff = alpha2 - beta2;

            const std::array<h_float, 3> sigmas = diagonal_sigma(current_state, shifted_k);
            const h_float xy_length = sqrt(sigmas[0]*sigmas[0] + sigmas[1]*sigmas[1]);
            alpha_beta_prod = 2 * sqrt(alpha2 * beta2) * ( is_zero(xy_length) ? 1.0 : sigmas[0] / xy_length );
            alpha_beta_imag = 2 * sqrt(alpha2 * beta2) * ( is_zero(xy_length) ? 0.0 : sigmas[1] / xy_length );

            z_epsilon = shifted_k.cos_z + dispersion(shifted_k);
            normalization = dispersion(shifted_k) * z_epsilon;
        
            relax_to_diagonal[0] = shifted_k.cos_x;
            relax_to_diagonal[1] = shifted_k.cos_y;
            relax_to_diagonal[2] = shifted_k.cos_z;
            for (auto& r : relax_to_diagonal) r *= (alpha_beta_diff * z_epsilon / normalization);
        
            relax_to_offdiagonal[0] = alpha_beta_prod * (  shifted_k.cos_y * shifted_k.cos_y + shifted_k.cos_z * z_epsilon);
            relax_to_offdiagonal[1] = alpha_beta_prod * (- shifted_k.cos_x * shifted_k.cos_y);
            relax_to_offdiagonal[2] = alpha_beta_prod * (- shifted_k.cos_x * z_epsilon);

            relax_to_offdiagonal[0] += alpha_beta_imag * (- shifted_k.cos_x * shifted_k.cos_y); 
            relax_to_offdiagonal[1] += alpha_beta_imag * (  shifted_k.cos_x * shifted_k.cos_x + shifted_k.cos_z * z_epsilon);
            relax_to_offdiagonal[2] += alpha_beta_imag * (- shifted_k.cos_y * z_epsilon);
            for (auto& r : relax_to_offdiagonal) r /= normalization;
        };

        update_equilibrium_state(laser->laser_function(time_config.t_begin));

        auto right_side = [this, &k, &laser, &prefactor, &update_equilibrium_state, &relax_to_diagonal, &relax_to_offdiagonal](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const sigma_state_type m = {k.cos_x, k.cos_y, std::cos(k.z - laser->laser_function(t))};
            update_equilibrium_state(laser->laser_function(t));
            dxdt = cross_product(m, state);
            for(size_t i = 0U; i < dxdt.size(); ++i) {
                dxdt[i] *= prefactor;
                dxdt[i] -= inverse_diagonal_relaxation_time * (state[i] / 3.0 - relax_to_diagonal[i]) // sigma^z
                    + inverse_offdiagonal_relaxation_time * (state[i] * (2. / 3.) - relax_to_offdiagonal[i]); // sigma^x and sigma^y -> therefore 2*state
            }
        };

        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;

        rhos.resize(time_config.n_measurements + 1);

#ifdef DDT_J
        constexpr double LASER_DT = 1e-5;
        sigma_state_type ddt_rho;
        right_side(current_state, ddt_rho, t_begin);
        rhos[0] = ddt_rho[2] * std::sin(k.z - laser->laser_function(t_begin)) 
            - current_state[2] * std::cos(k.z - laser->laser_function(t_begin)) 
            * (laser->laser_function(t_begin + LASER_DT) - laser->laser_function(t_begin - LASER_DT)) / (2 * LASER_DT);
#else
        rhos[0] = current_state[2];
#endif

        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
#ifdef DDT_J
            right_side(current_state, ddt_rho, t_end);
            rhos[i] = ddt_rho[2] * std::sin(k.z - laser->laser_function(t_end)) 
                - current_state[2] * std::cos(k.z - laser->laser_function(t_end)) 
                * (laser->laser_function(t_end + LASER_DT) - laser->laser_function(t_end - LASER_DT)) / (2 * LASER_DT);
#else
            rhos[i] = current_state[2];
#endif
            t_begin = t_end;
            t_end += measure_every;
        }
    }

    void PiFlux::evolve_occupation_numbers(std::vector<OccupationContainer::occupation_t>& occupations, Laser::Laser const * const laser, 
            const momentum_type& k, const TimeIntegrationConfig& time_config, bool diagonal /*= true*/) const
    {
        auto compute_occupations = [this, diagonal](sigma_state_type const& input, momentum_type const& k) {
            if (diagonal) {
                const auto diags = diagonal_sigma(input, k);
                return OccupationContainer::occupation_t{ 
                    std::sqrt( 
                        0.5 * ( -diags[2] + norm(diags[0], diags[1], diags[2] ) )
                    ),
                    std::sqrt( 
                        0.5 * ( diags[2] + norm(diags[0], diags[1], diags[2] ) )
                    )
                };
            }
            return OccupationContainer::occupation_t{ 
                std::sqrt( 
                    0.5 * ( -input[2] + norm(input[0], input[1], input[2] ) )
                ),
                std::sqrt( 
                    0.5 * ( input[2] + norm(input[0], input[1], input[2] ) )
                )
            };
        };
        const h_float prefactor = 4. * hopping_element;

        momentum_type shifted_k = k;

        h_float alpha2 = occupation_a(k);
        h_float beta2 = occupation_b(k);
        h_float alpha_beta_diff = alpha2 - beta2;
        h_float alpha_beta_prod = 2 * sqrt(alpha2 * beta2);
        h_float alpha_beta_imag{};
        h_float z_epsilon = k.cos_z + dispersion(k);
        h_float normalization = dispersion(k) * z_epsilon;
            
        sigma_state_type relax_to_diagonal;
        sigma_state_type relax_to_offdiagonal; 
        sigma_state_type current_state = ic_sigma(k, alpha_beta_diff, alpha_beta_prod, z_epsilon);
            
        auto update_equilibrium_state = [&](const h_float laser_at_t, const h_float __t) {
            shifted_k.update_z(k.z - laser_at_t);

            if (is_zero(shifted_k.cos_x) && is_zero(shifted_k.cos_y) && shifted_k.cos_z < h_float{}) {
                shifted_k.cos_x = std::abs(shifted_k.cos_z) * RESCUE_TRAFO;
                shifted_k.cos_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            
            alpha2 = occupation_a(shifted_k);
            beta2 = occupation_b(shifted_k);
            alpha_beta_diff = alpha2 - beta2;

            const std::array<h_float, 3> sigmas = diagonal_sigma(current_state, shifted_k);
            const h_float xy_length = sqrt(sigmas[0]*sigmas[0] + sigmas[1]*sigmas[1]);
            alpha_beta_prod = 2 * sqrt(alpha2 * beta2) * ( is_zero(xy_length) ? 1.0 : sigmas[0] / xy_length );
            alpha_beta_imag = 2 * sqrt(alpha2 * beta2) * ( is_zero(xy_length) ? 0.0 : sigmas[1] / xy_length );

            z_epsilon = shifted_k.cos_z + dispersion(shifted_k);
            normalization = dispersion(shifted_k) * z_epsilon;
            
            relax_to_diagonal[0] = shifted_k.cos_x;
            relax_to_diagonal[1] = shifted_k.cos_y;
            relax_to_diagonal[2] = shifted_k.cos_z;
            for (auto& r : relax_to_diagonal) r *= (alpha_beta_diff * z_epsilon / normalization);
        
            relax_to_offdiagonal[0] = alpha_beta_prod * (  shifted_k.cos_y * shifted_k.cos_y + shifted_k.cos_z * z_epsilon);
            relax_to_offdiagonal[1] = alpha_beta_prod * (- shifted_k.cos_x * shifted_k.cos_y);
            relax_to_offdiagonal[2] = alpha_beta_prod * (- shifted_k.cos_x * z_epsilon);

            relax_to_offdiagonal[0] += alpha_beta_imag * (- shifted_k.cos_x * shifted_k.cos_y); 
            relax_to_offdiagonal[1] += alpha_beta_imag * (  shifted_k.cos_x * shifted_k.cos_x + shifted_k.cos_z * z_epsilon);
            relax_to_offdiagonal[2] += alpha_beta_imag * (- shifted_k.cos_y * z_epsilon);
            for (auto& r : relax_to_offdiagonal) r /= normalization;
        };

        update_equilibrium_state(laser->laser_function(time_config.t_begin), 0.0);

        auto right_side = [this, &k, &laser, &prefactor, &update_equilibrium_state, &relax_to_diagonal, &relax_to_offdiagonal](const sigma_state_type& state, sigma_state_type& dxdt, const h_float t) {
            const sigma_state_type m = {k.cos_x, k.cos_y, std::cos(k.z - laser->laser_function(t))};
            update_equilibrium_state(laser->laser_function(t), t);
            dxdt = cross_product(m, state);
            for(size_t i = 0U; i < dxdt.size(); ++i) {
                dxdt[i] *= prefactor;
                dxdt[i] -= inverse_diagonal_relaxation_time * (state[i] / 3.0 - relax_to_diagonal[i]) // sigma^z
                    + inverse_offdiagonal_relaxation_time * (state[i] * (2. / 3.) - relax_to_offdiagonal[i]); // sigma^x and sigma^y -> therefore 2*state
            }
        };
        
        const h_float measure_every = time_config.measure_every();
        const h_float dt = time_config.dt();
        h_float t_begin = time_config.t_begin;
        h_float t_end = t_begin + measure_every;
    
        occupations[0] = compute_occupations(current_state, shifted_k);
        for (int i = 1; i <= time_config.n_measurements; ++i) {
            integrate_adaptive(make_controlled<sigma_error_stepper_type>(abs_error, rel_error), right_side, current_state, t_begin, t_end, dt);
            occupations[i] = compute_occupations(current_state, shifted_k);
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

    std::string PiFlux::get_property_in_SI_units(const std::string& property, const h_float photon_energy) const
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

    h_float PiFlux::occupation_a(const momentum_type& k) const 
    {
        return fermi_function(-E_F + 2 * hopping_element * dispersion(k), beta);
    }

    h_float PiFlux::occupation_b(const momentum_type& k) const
    {
        return fermi_function(-E_F - 2 * hopping_element * dispersion(k), beta);
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
        if (inverse_diagonal_relaxation_time > h_float{}) {
            return time_evolution_diagonal_relaxation(rhos, laser, k, time_config);
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

    PiFlux::sigma_state_type PiFlux::ic_sigma(const momentum_type &k, h_float alpha_beta_diff, h_float alpha_beta_prod, h_float z_epsilon) const noexcept
    {
        return sigma_state_type{ic_sigma_x(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_y(k, alpha_beta_diff, alpha_beta_prod, z_epsilon), 
            ic_sigma_z(k, alpha_beta_diff, alpha_beta_prod, z_epsilon)};
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
            if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
                k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
                k.cos_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }

            x_buffer = improved_xy_integral(k, rhos_buffer, laser, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                x_buffer[i] *= z_gauss::weights[z]
#ifndef DDT_J
                    * std::sin(k.z - laser->laser_function(i * time_step + time_config.t_begin))
#endif
                ;
            }
            std::transform(current_density_time.begin(), current_density_time.end(), x_buffer.begin(), current_density_time.begin(), std::plus<>());
        
            /*
            *  -z
            */
            k.update_z(-pi * z_gauss::abscissa[z]);
            if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
                k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
                k.cos_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }

            x_buffer = improved_xy_integral(k, rhos_buffer, laser, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                x_buffer[i] *= z_gauss::weights[z]
#ifndef DDT_J
                    * std::sin(k.z - laser->laser_function(i * time_step + time_config.t_begin))
#endif
                ;
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

    std::vector<OccupationContainer> PiFlux::compute_occupation_numbers(Laser::Laser const * const laser, 
        TimeIntegrationConfig const& time_config, const int N) const
    {
        const h_float dx = pi / N;
        const h_float dz = pi / N;
        
        auto k_xy = [dx](int x) {
            return x * dx;
        };
        auto k_z = [dz](int z) {
            return dz * z;
        };

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
            PROGRESS_BAR_UPDATE(N*N);

            const int x = i / N;
            const int z = i % N;
            momentum_type k(k_xy(x), 0.5 * pi, k_z(z));
            if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
                k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
                k.cos_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
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

    std::array<std::vector<h_float>, 2> PiFlux::current_per_energy(Laser::Laser const * const laser, 
            TimeIntegrationConfig const& time_config, const int N) const
    {
        constexpr h_float energy_cut = 0.5 * sqrt_3;

        const h_float dxy = pi / N;
        const h_float dz = pi / N;
        
        auto k_xy = [dxy](int x) {
            return x * dxy;
        };
        auto k_z = [dz](int z) {
            return dz * z;
        };

        const h_float time_step = time_config.measure_every();

        nd_vector rhos_buffer;

        std::vector<h_float> current_density_non_dirac(time_config.n_measurements + 1, h_float{});
        std::vector<h_float> current_density_dirac(time_config.n_measurements + 1, h_float{});
        
#ifndef MROCK_CL1_CASCADE
        std::vector<int> progresses(omp_get_max_threads(), int{});
#endif
#pragma omp parallel for private(rhos_buffer) reduction(vec_plus:current_density_dirac,current_density_non_dirac) schedule(dynamic)
        for (int i = 0; i < N * N * N; ++i) {
            PROGRESS_BAR_UPDATE(N*N*N);

            const int z = i / (N * N);
            const int y = (i / N) % N;
            const int x = i % N;

            momentum_type k(k_xy(x), k_xy(y), k_z(z));
            if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
                k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
                k.cos_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            
            if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
                k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
                k.cos_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            __time_evolution__(rhos_buffer, laser, k, time_config);

            for (int i = 0; i <= time_config.n_measurements; ++i) {
                const h_float laser_value = laser->laser_function(i * time_step + time_config.t_begin);
                const h_float instantaneous_energy = dispersion(k.shift_z(laser_value));

                if (instantaneous_energy < energy_cut) {
                    current_density_dirac[i] += rhos_buffer(i) * std::sin(k.z - laser_value);
                }
                else {
                    current_density_non_dirac[i] += rhos_buffer(i) * std::sin(k.z - laser_value);
                }
            }
            

            /*
            *  -z
            */
            k.invert();
            if (is_zero(k.cos_x) && is_zero(k.cos_y) && k.cos_z < h_float{}) {
                k.cos_x = std::abs(k.cos_z) * RESCUE_TRAFO;
                k.cos_z *= (1. - 0.5*RESCUE_TRAFO*RESCUE_TRAFO);
            }
            __time_evolution__(rhos_buffer, laser, k, time_config);
            for (int i = 0; i <= time_config.n_measurements; ++i) {
                const h_float laser_value = laser->laser_function(i * time_step + time_config.t_begin);
                const h_float instantaneous_energy = dispersion(k.shift_z(laser_value));

                if (instantaneous_energy < energy_cut) {
                    current_density_dirac[i] += rhos_buffer(i) * std::sin(k.z - laser_value);
                }
                else {
                    current_density_non_dirac[i] += rhos_buffer(i) * std::sin(k.z - laser_value);
                }
            }
        }

        for (auto& val : current_density_dirac) {
            val /= N*N*N;
        }
        for (auto& val : current_density_non_dirac) {
            val /= N*N*N;
        }
        return {current_density_dirac, current_density_non_dirac};
    }

    std::array<h_float, 3> PiFlux::diagonal_sigma(sigma_state_type const& input, momentum_type const& k) const
    {
        const h_float z = k.cos_z + dispersion(k);
        const h_float norm = 2 * dispersion(k) * z;
        const h_float __xz = 2 * k.cos_x * z;
        const h_float __yz = 2 * k.cos_y * z;
        const h_float __xy = 2 * k.cos_x * k.cos_y;

        return { 
            (input[0] * (z*z - k.cos_x*k.cos_x + k.cos_y*k.cos_y) - __xy * input[1] - __xz * input[2]) / norm,
            (input[1] * (z*z + k.cos_x*k.cos_x - k.cos_y*k.cos_y) - __xy * input[0] - __yz * input[2]) / norm,
            (input[2] * (z*z - k.cos_x*k.cos_x - k.cos_y*k.cos_y) + __xz * input[0] + __yz * input[1]) / norm 
        };
    }
}