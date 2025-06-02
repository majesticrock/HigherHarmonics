#pragma once
#include "Dispatcher.hpp"
#include <mrock/utility/InputFileReader.hpp>
#include "../Systems/Honeycomb.hpp"

namespace HHG::Dispatch {
    struct HoneycombDispatcher : public Dispatcher {
        Systems::Honeycomb system;
        h_float photon_energy;

        HoneycombDispatcher(mrock::utility::InputFileReader& input, int N, h_float t0_offset = h_float{});

        void compute(int rank, int n_ranks, int n_z) final;
        void debug(int n_z) final;

        virtual nlohmann::json special_information() const override;
    };
}