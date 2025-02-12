#pragma once

#include <mrock/utility/InputFileReader.hpp>
#include "GlobalDefinitions.hpp"

namespace hgg {
    class DiracSystem {
        h_float E_F{};
        h_float v_F{};
        h_float band_width{};
        h_float max_k{};
    public:
        DiracSystem() = default;
        DiracSystem(mrock::utility::InputFileReader& input);

        inline h_float dispersion(h_float k) {
            return (v_F * k + E_F);
        }
    };
}