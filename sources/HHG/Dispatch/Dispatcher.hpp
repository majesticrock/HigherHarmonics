#pragma once
#include "../GlobalDefinitions.hpp"
#include "../TimeIntegrationConfig.hpp"
#include "../Laser/Laser.hpp"

#include <mrock/utility/InputFileReader.hpp>
#include <array>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
#include <string_view>

namespace HHG::Dispatch {
    struct Dispatcher {
        h_float diagonal_relaxation_time{};

        std::vector<h_float> current_density_time;
        std::array<std::vector<h_float>, n_debug_points> time_evolutions;
        std::unique_ptr<Laser::Laser> laser;
        TimeIntegrationConfig time_config;

        virtual ~Dispatcher() = default;
        Dispatcher(int N, h_float _diagonal_relaxation_time) 
            : diagonal_relaxation_time(_diagonal_relaxation_time), current_density_time(N + 1)
        {
            time_evolutions.fill(std::vector<h_float>(N + 1));
        }

        virtual void compute(int rank, int n_ranks, int n_z) = 0;
        virtual void debug(int n_z) = 0;

        virtual nlohmann::json special_information() const;

        static h_float get_photon_energy(mrock::utility::InputFileReader &input);

        static constexpr std::string_view continuous    = "continuous";
        static constexpr std::string_view cosine        = "cosine";
        static constexpr std::string_view quench        = "quench";
        static constexpr std::string_view powerlaw      = "powerlaw";
        static constexpr std::string_view exp           = "exp";
        static constexpr std::string_view expA          = "expA";
        static constexpr std::string_view expB          = "expB";
        static constexpr std::string_view dcos          = "dcos";
        static constexpr std::string_view dcosA         = "dcosA";
        static constexpr std::string_view dcosB         = "dcosB";
        static constexpr std::string_view gauss         = "gauss";
        static constexpr std::string_view dgauss        = "dgauss";
        static constexpr std::string_view dgaussA       = "dgaussA";
        static constexpr std::string_view dgaussB       = "dgaussB";
    };
}