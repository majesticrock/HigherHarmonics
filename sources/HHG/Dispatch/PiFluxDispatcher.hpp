#pragma once
#include "Dispatcher.hpp"
#include <mrock/utility/InputFileReader.hpp>
#include "../Systems/PiFlux.hpp"

namespace HHG::Dispatch {
    struct PiFluxDispatcher : public Dispatcher {
        Systems::PiFlux system;

        PiFluxDispatcher(mrock::utility::InputFileReader& input, int N, h_float t0_offset = h_float{});

        void compute(int rank, int n_ranks, int n_z) final;
        void debug(int n_z) final;
    };
}