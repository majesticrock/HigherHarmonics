#pragma once
#include "Dispatcher.hpp"
#include <mrock/utility/InputFileReader.hpp>
#include "../Systems/PiFlux.hpp"
#include "../Systems/ModifiedPiFlux.hpp"
#include <variant>

namespace HHG::Dispatch {
    struct PiFluxDispatcher : public Dispatcher {
        bool mod_system{};
        std::variant<Systems::PiFlux, Systems::ModifiedPiFlux> system;
        h_float photon_energy;
        

        PiFluxDispatcher(mrock::utility::InputFileReader& input, int N, h_float t0_offset = h_float{});
        
        void compute(int rank, int n_ranks, int n_z) final;
        void debug(int n_z) final;

        std::vector<OccupationContainer> track_occupation_numbers(int N) const;

        std::array<std::vector<h_float>, 2> compute_split_current(int N) const;

        virtual nlohmann::json special_information() const override;
    };
}