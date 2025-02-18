#pragma once
#include "GlobalDefinitions.hpp"
#include "Laser.hpp"
#include "TimeIntegrationConfig.hpp"

#include <functional>

namespace HHG {
    struct GreensFunction {
        using c_vector = std::vector<h_complex>;
        using r_vector = std::vector<h_float>;
        using time_evolution_function = std::function<void(c_vector&, c_vector&, Laser const * const, h_float, h_float, const TimeIntegrationConfig&)>;
        
        c_vector alphas;
        c_vector betas;
        c_vector time_domain_greens_function;
        c_vector fourier_greens_function;

        const time_evolution_function time_evolution;
        const int N;
        const int measurements_per_cycle;
        const int greens_N;

        template<class F>
        GreensFunction(const F& complex_time_evolution,int _N, int _measurements_per_cycle)
            : time_evolution(complex_time_evolution), N{_N}, 
            measurements_per_cycle{_measurements_per_cycle},
            greens_N{N / 2 - measurements_per_cycle}
        {}

        void compute_alphas_betas(Laser const * const laser, h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config);

        void compute_time_domain_greens_function(const int t_center, h_float phase_factor = 0);

        void fourier_transform_greens_function();

        r_vector get_alphas_real() const;
        r_vector get_alphas_imag() const;

        r_vector get_betas_real() const;
        r_vector get_betas_imag() const; 

        r_vector get_time_domain_greens_function_real() const;
        r_vector get_time_domain_greens_function_imag() const; 

        r_vector get_fourier_greens_function_real() const;
        r_vector get_fourier_greens_function_imag() const; 
    private:
        inline h_complex G(int t_rel, int t_ave) const {
            if (t_rel < 0) {
                return h_complex{};
            }
            const int offset = t_ave + N / 2;
            const int t1 = offset + t_rel;
            const int t2 = offset - t_rel;

            if (t1 >= N || t2  >= N || t1 < 0 || t2 < 0) {
                return h_complex{};
            }
            return -imaginary_unit * (std::conj(alphas[t1]) * alphas[t2] + std::conj(betas[t1]) * betas[t2]);
        }
    };
}