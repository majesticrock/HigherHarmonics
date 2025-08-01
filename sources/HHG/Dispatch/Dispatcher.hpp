#pragma once
#include "../GlobalDefinitions.hpp"
#include "../TimeIntegrationConfig.hpp"
#include "../Laser/Laser.hpp"

#include <mrock/utility/InputFileReader.hpp>
#include <array>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace HHG::Dispatch {
    struct Dispatcher {
        h_float diagonal_relaxation_time{};

        std::vector<h_float> current_density_time;
        std::array<std::vector<h_float>, n_debug_points> time_evolutions;
        std::unique_ptr<Laser::Laser> laser;
        TimeIntegrationConfig time_config;

        virtual ~Dispatcher() = default;
        Dispatcher(int N, h_float _diagonal_relaxation_time) : diagonal_relaxation_time(_diagonal_relaxation_time), current_density_time(N + 1)
        {
            time_evolutions.fill(std::vector<h_float>(N + 1));
        }

        virtual void compute(int rank, int n_ranks, int n_z) = 0;
        virtual void debug(int n_z) = 0;

        virtual nlohmann::json special_information() const;

        static h_float get_photon_energy(mrock::utility::InputFileReader &input);
    };
}