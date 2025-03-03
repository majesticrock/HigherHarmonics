#pragma once
#include "../GlobalDefinitions.hpp"
#include "../TimeIntegrationConfig.hpp"

namespace HHG::Fourier {
    struct FourierIntegral {
        h_float delta_t;
        std::vector<h_float> frequencies;
        std::vector<h_float> time_samples;

        FourierIntegral() = delete;
        FourierIntegral(TimeIntegrationConfig const& time_config);

        void compute(const std::vector<h_float>& input, std::vector<h_float>& real_output, std::vector<h_float>& imag_output) const;
    };
}