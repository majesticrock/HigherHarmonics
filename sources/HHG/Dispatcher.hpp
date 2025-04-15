#pragma once
#include "GlobalDefinitions.hpp"
#include <array>
#include <vector>
#include <memory>

#include "TimeIntegrationConfig.hpp"
#include "Laser/Laser.hpp"

namespace HHG {
    struct Dispatcher {
        h_float decay_time{};

        std::vector<h_float> current_density_time;
        std::array<std::vector<h_float>, n_debug_points> time_evolutions;
        std::unique_ptr<Laser::Laser> laser;
        TimeIntegrationConfig time_config;

        Dispatcher(int N, h_float _decay_time) : decay_time(_decay_time), current_density_time(N)
        {
            time_evolutions.fill(std::vector<h_float>(N));
        }

        virtual void compute(int rank, int n_ranks, int n_z) = 0;
        virtual void debug(int n_z) = 0;
    };
}