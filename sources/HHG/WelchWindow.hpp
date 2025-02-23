#pragma once
#include "GlobalDefinitions.hpp"
#include <mrock/utility/ConstexprPower.hpp>

namespace HHG {
    struct WelchWindow {
        const int N;

        WelchWindow() = delete;
        constexpr WelchWindow(int _N) : N(_N) {}

        constexpr h_float operator[](int i) const {
            return 1. - mrock::utility::constexprPower<2, h_float, h_float>((i - 0.5 * N) / (0.5 * N));
        }
    };
}