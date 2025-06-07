#pragma once
#include "../GlobalDefinitions.hpp"
#include "../TimeIntegrationConfig.hpp"
#include "../Laser/Laser.hpp"

#include <array>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace HHG::Dispatch {
    struct Dispatcher {
        h_float decay_time{};

        std::vector<h_float> current_density_time;
        std::array<std::vector<h_float>, n_debug_points> time_evolutions;
        std::unique_ptr<Laser::Laser> laser;
        TimeIntegrationConfig time_config;

        virtual ~Dispatcher() = default;
        Dispatcher(int N, h_float _decay_time) : decay_time(_decay_time), current_density_time(N + 1)
        {
            time_evolutions.fill(std::vector<h_float>(N + 1));
        }

        virtual void compute(int rank, int n_ranks, int n_z) = 0;
        virtual void debug(int n_z) = 0;

        virtual nlohmann::json special_information() const;
    };
}