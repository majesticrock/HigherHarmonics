#pragma once

#include "GlobalDefinitions.hpp"

namespace HHG {
    struct TimeIntegrationConfig {
        h_float t_begin{};
        h_float t_end{};
        int n_measurements{};
        int n_subdivisions{};

        h_float dt() const;
        h_float measure_every() const;
    };
}