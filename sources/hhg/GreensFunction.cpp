#include "GreensFunction.hpp"
#include "ComplexFFT.hpp"

#include <mrock/utility/FunctionTime.hpp>
#include <mrock/utility/ComplexNumberIterators.hpp>

#include <iostream>

namespace HHG {
    inline GreensFunction::r_vector c_vector_to_real(const GreensFunction::c_vector& in, const int N) {
        auto real_begin = mrock::utility::make_real_part_iterator(in.data());
        auto real_end = mrock::utility::make_real_part_iterator_end(in.data(), N);
        return GreensFunction::r_vector(real_begin, real_end);
    }
    inline GreensFunction::r_vector c_vector_to_imag(const GreensFunction::c_vector& in, const int N) {
        auto imag_begin = mrock::utility::make_imag_part_iterator(in.data());
        auto imag_end = mrock::utility::make_imag_part_iterator_end(in.data(), N);
        return GreensFunction::r_vector(imag_begin, imag_end);
    }

    void GreensFunction::compute_alphas_betas(Laser const * const laser, h_float k_z, h_float kappa, const TimeIntegrationConfig& time_config)
    {
        std::cout << "Computing alphas and betas for the Green's function." << std::endl;
        mrock::utility::function_time_ms(time_evolution, alphas, betas, laser, k_z, kappa, time_config);
    }

    void GreensFunction::compute_time_domain_greens_function(const int t_center, h_float phase_factor /* = 0 */ )
    {
        using std::chrono::high_resolution_clock;

        std::cout << "Computing laser-cycle average for the Green's function." << std::endl;
        high_resolution_clock::time_point begin = high_resolution_clock::now();

        time_domain_greens_function.resize(greens_N, h_complex{});
        for (int t_rel = 0; t_rel < greens_N; ++t_rel) {
            for (int i = - measurements_per_cycle / 2 + 1; i < measurements_per_cycle; ++i) {
                time_domain_greens_function[t_rel] += G(t_rel, i + t_center);
            }
            time_domain_greens_function[t_rel] += 0.5 * (G(t_rel, t_center) + G(t_rel, t_center + measurements_per_cycle));
            time_domain_greens_function[t_rel] /= static_cast<h_float>(measurements_per_cycle);
            time_domain_greens_function[t_rel] *= std::exp(-imaginary_unit * (2  * phase_factor * t_rel));
            //const h_float x = 4 * static_cast<h_float>(t_rel) / static_cast<h_float>(greens_N);
            //time_domain_greens_function[t_rel] *= std::exp(-x);
        }

        high_resolution_clock::time_point end = high_resolution_clock::now();
		std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }

    void GreensFunction::fourier_transform_greens_function()
    {
        using std::chrono::high_resolution_clock;
        std::cout << "Computing F-trafo for the Green's function." << std::endl;
        high_resolution_clock::time_point begin = high_resolution_clock::now();

        ComplexFFT fft(greens_N);
        fft.compute(time_domain_greens_function, fourier_greens_function);

        high_resolution_clock::time_point end = high_resolution_clock::now();
		std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }

    GreensFunction::r_vector GreensFunction::get_alphas_real() const
    {
        return c_vector_to_real(alphas, N + 1);
    }
    GreensFunction::r_vector GreensFunction::get_alphas_imag() const
    {
        return c_vector_to_imag(alphas, N + 1);
    }

    GreensFunction::r_vector GreensFunction::get_betas_real() const
    {
        return c_vector_to_real(betas, N + 1);
    }
    GreensFunction::r_vector GreensFunction::get_betas_imag() const
    {
        return c_vector_to_imag(betas, N + 1);
    }

    GreensFunction::r_vector GreensFunction::get_time_domain_greens_function_real() const
    {
        return c_vector_to_real(time_domain_greens_function, greens_N);
    }
    GreensFunction::r_vector GreensFunction::get_time_domain_greens_function_imag() const
    {
        return c_vector_to_imag(time_domain_greens_function, greens_N);
    }

    GreensFunction::r_vector GreensFunction::get_fourier_greens_function_real() const
    {
        return c_vector_to_real(fourier_greens_function, greens_N);
    }
    GreensFunction::r_vector GreensFunction::get_fourier_greens_function_imag() const
    {
        return c_vector_to_imag(fourier_greens_function, greens_N);
    }
}