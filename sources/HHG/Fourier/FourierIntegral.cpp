#include "FourierIntegral.hpp"
#include <cmath>
#include <cassert>
#include <cstddef>

namespace HHG::Fourier {
    FourierIntegral::FourierIntegral(TimeIntegrationConfig const& time_config)
        : delta_t{time_config.measure_every()}, frequencies(time_config.n_measurements / 2 + 1), time_samples(time_config.n_measurements + 1)
    {
        const h_float omega_max = 64;
        const h_float delta_omega = omega_max / frequencies.size();
        for (std::size_t i = 0U; i < frequencies.size(); ++i) {
            frequencies[i] = i * delta_omega;
        }
        for (std::size_t  i = 0U; i < time_samples.size(); ++i) {
            time_samples[i] = time_config.t_begin + i * delta_t;
        }
    }

    void FourierIntegral::compute(const std::vector<h_float>& input, std::vector<h_float>& real_output, std::vector<h_float>& imag_output) const
    {
        assert(input.size() >= time_samples.size());
        real_output.resize(frequencies.size());
        imag_output.resize(frequencies.size());

        for (std::size_t  i = 0U; i < frequencies.size(); ++i) {
            real_output[i] = 0.5 * (
                std::cos(frequencies[i] * time_samples[0]) * input[0] + std::cos(frequencies[i] * time_samples.back()) * input.back()
            );
            imag_output[i] = 0.5 * (
                std::sin(frequencies[i] * time_samples[0]) * input[0] + std::sin(frequencies[i] * time_samples.back()) * input.back()
            );

            for (std::size_t  t = 1U; t < time_samples.size() - 1; ++t) {
                real_output[i] += std::cos(frequencies[i] * time_samples[t]) * input[t];
                imag_output[i] += std::sin(frequencies[i] * time_samples[t]) * input[t];
            }
            real_output[i] *= delta_t;
            imag_output[i] *= delta_t;
        }
    }
}