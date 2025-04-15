#include "PiFlux.hpp"
#include "GeneralMagnus.hpp"
#include "Laser/gauss.hpp"

#include <cmath>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <nlohmann/json.hpp>

#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/progress_bar.hpp>

typedef Eigen::Vector<HHG::h_float, 3> sigma_state_type;

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

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

namespace HHG {
    PiFlux::PiFlux(h_float temperature, h_float _E_F, h_float _v_F, h_float _band_width, h_float _photon_energy)
        : beta(is_zero(temperature) ? std::numeric_limits<h_float>::infinity() : _photon_energy / (k_B * temperature)), 
            E_F(_E_F / _photon_energy), 
            hopping_element(_band_width / sqrt_12), 
            lattice_constant(sqrt_3 * hbar * _v_F / (_photon_energy * _band_width))
    { }

    void PiFlux::time_evolution_magnus(nd_vector &rhos, Laser::Laser const *const laser, const momentum_type& k, const TimeIntegrationConfig &time_config) const
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

    std::vector<h_float> PiFlux::compute_current_density(Laser::Laser const *const laser, TimeIntegrationConfig const &time_config, const int rank, const int n_ranks, const int n_z) const
    {
        nd_vector rhos_buffer = nd_vector::Zero(time_config.n_measurements + 1);
        std::vector<h_float> current_density_time(time_config.n_measurements + 1, h_float{});

        const h_float momentum_ratio = pi / n_z;

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

            // k_z = 0 and k_z = \pm pi do not matter due to the sin(k_z) factor in the current density
            if (z == n_z/2) continue;

            momentum_type k;
            k.update_z(z * momentum_ratio - pi);

            nd_vector x_buffer = nd_vector::Zero(time_config.n_measurements + 1); 
            // At k_x = k_y = \pm pi (that would be x=y=\pm n_z / 2), we have nothing but a rotation around the z-axis, 
            // therefore sigma_z = const.
            // We also have the symmetry that rho(k_x) = rho(-k_x) as everything depends merely on cos(k_x).
            // The same applies to k_y, but not to k_z.
            for (int x = 1; x < n_z / 2; ++x) {
                k.update_x(x * 0.5 * momentum_ratio);

                nd_vector y_buffer = nd_vector::Zero(time_config.n_measurements + 1);
                for (int y = 1; y < n_z / 2; ++y) {
                    k.update_y(y * 0.5 * momentum_ratio);
                    time_evolution_magnus(rhos_buffer, laser, k, time_config);
                    rhos_buffer /= dispersion(k);
                    y_buffer += rhos_buffer;
                }
                y_buffer *= 2.0;

                k.update_y(0.0);
                time_evolution_magnus(rhos_buffer, laser, k, time_config);
                rhos_buffer /= dispersion(k);
                y_buffer += rhos_buffer;

                k.update_y(pi / 2);
                time_evolution_magnus(rhos_buffer, laser, k, time_config);
                rhos_buffer /= dispersion(k);
                y_buffer += rhos_buffer;

                x_buffer += y_buffer;
            }
            x_buffer *= 2.0;

            {
                k.update_x(0.0);

                nd_vector y_buffer = nd_vector::Zero(time_config.n_measurements + 1);
                for (int y = 1; y < n_z / 2; ++y) {
                    k.update_y(y * 0.5 * momentum_ratio);
                    time_evolution_magnus(rhos_buffer, laser, k, time_config);
                    rhos_buffer /= dispersion(k);
                    y_buffer += rhos_buffer;
                }
                y_buffer *= 2.0;

                k.update_y(0.0);
                time_evolution_magnus(rhos_buffer, laser, k, time_config);
                rhos_buffer /= dispersion(k);
                y_buffer += rhos_buffer;

                k.update_y(pi / 2);
                time_evolution_magnus(rhos_buffer, laser, k, time_config);
                rhos_buffer /= dispersion(k);
                y_buffer += rhos_buffer;

                x_buffer += y_buffer;
            }
            {
                k.update_x(pi / 2);

                nd_vector y_buffer = nd_vector::Zero(time_config.n_measurements + 1);
                for (int y = 1; y < n_z / 2; ++y) {
                    k.update_y(y * 0.5 * momentum_ratio);
                    time_evolution_magnus(rhos_buffer, laser, k, time_config);
                    rhos_buffer /= dispersion(k);
                    y_buffer += rhos_buffer;
                }
                y_buffer *= 2.0;

                k.update_y(0.0);
                time_evolution_magnus(rhos_buffer, laser, k, time_config);
                rhos_buffer /= dispersion(k);
                y_buffer += rhos_buffer;

                x_buffer += y_buffer;
            }

            x_buffer *= k.cos_z * std::sin(k.z);
            std::transform(current_density_time.begin(), current_density_time.end(), x_buffer.begin(), current_density_time.begin(), std::plus<>());
        }
        std::cout << std::endl;

        for (auto& j : current_density_time) {
            j /= (n_z * n_z * n_z);
        }

        return current_density_time;
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
}