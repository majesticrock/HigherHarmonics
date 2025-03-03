#pragma once
#include "../GlobalDefinitions.hpp"

namespace HHG::Fourier {
    struct WelchWindow {
        const int N;

        WelchWindow() = delete;
        constexpr WelchWindow(int _N) : N(_N) {}

        constexpr h_float operator[](int i) const {
            return 1. - (i - 0.5 * N) * (i - 0.5 * N) / (0.25 * N * N);
        }
    };
}