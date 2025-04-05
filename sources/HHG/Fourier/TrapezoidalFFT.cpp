#include "TrapezoidalFFT.hpp"
#include <cmath>
#include "FFT.hpp"

namespace HHG::Fourier {
    TrapezoidalFFT::TrapezoidalFFT(TimeIntegrationConfig const &time_config)
        : delta_t(time_config.measure_every()), t_begin(time_config.t_begin), t_end(time_config.t_end)
    { }

    void TrapezoidalFFT::compute(const std::vector<h_float> &input, std::vector<h_float> &real_output, std::vector<h_float> &imag_output)
    {
        const int n_input = input.size();
        const int n_fft = n_input / 2 + 1;
        // theta = time_step * omega
        auto W = [](h_float theta) -> h_float {
            if (is_zero(theta)) {
                return 1.;
            }
            return 2. * (1. - std::cos(theta)) / (theta * theta);
        };
        auto alpha_0 = [](h_float theta) -> h_complex {
            if (is_zero(theta)) {
                return -0.5;
            }
            return (-(1. - std::cos(theta)) + imaginary_unit * (theta - std::sin(theta))) / (theta * theta);
        };

        real_output.resize(n_fft);
        imag_output.resize(n_fft);
        FFT fft(n_input);
        fft.compute(input, real_output, imag_output);

        frequencies.resize(n_fft);
        for (int i = 0; i < n_fft; ++i) {
            frequencies[i] = 2 * i * pi / (n_input * delta_t);
            const h_float theta = frequencies[i] * delta_t; // 2 pi is contained within the unit of the time
            const h_complex corrections = alpha_0(theta) * input[0] + std::conj(alpha_0(theta)) * std::polar(1.0, frequencies[i] * (t_end - t_begin)) * input[n_input - 1];
            const h_complex corrected_output = std::polar(delta_t * W(theta), frequencies[i] * t_begin) * h_complex(real_output[i], imag_output[i]) + corrections;

            real_output[i] = corrected_output.real();
            imag_output[i] = corrected_output.imag();
        }
    }
}