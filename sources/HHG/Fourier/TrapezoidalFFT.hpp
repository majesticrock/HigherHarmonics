#pragma once
#include "../GlobalDefinitions.hpp"
#include "../TimeIntegrationConfig.hpp"

namespace HHG::Fourier {
    struct TrapezoidalFFT {
        h_float delta_t;
        h_float t_begin;
        h_float t_end;
        std::vector<h_float> frequencies;

        TrapezoidalFFT() = delete;
        TrapezoidalFFT(TimeIntegrationConfig const& time_config);

        void compute(const std::vector<h_float>& input, std::vector<h_float>& real_output, std::vector<h_float>& imag_output);
    };
}