#pragma once
#include "GlobalDefinitions.hpp"
#include <array>
#include <vector>
#include <memory>

#include "TimeIntegrationConfig.hpp"
#include "Laser/Laser.hpp"

namespace HHG {
    struct Dispatcher {
        std::vector<h_float> current_density_time;
        std::array<std::vector<h_float>, n_debug_points> time_evolutions;
        std::unique_ptr<Laser::Laser> laser;
        TimeIntegrationConfig time_config;

        Dispatcher(int N) : current_density_time(N)
        {
            time_evolutions.fill(std::vector<h_float>(N));
        }

        virtual void compute() = 0;
        virtual void debug() = 0;
    };
}