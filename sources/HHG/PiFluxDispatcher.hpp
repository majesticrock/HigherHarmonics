#pragma once
#include "Dispatcher.hpp"
#include <mrock/utility/InputFileReader.hpp>
#include "PiFlux.hpp"

namespace HHG {
    struct PiFluxDispatcher : public Dispatcher {
        PiFlux system;

        PiFluxDispatcher(mrock::utility::InputFileReader& input, int N, h_float t0_offset = h_float{});

        void compute(int rank, int n_ranks, int n_z) final;
        void debug(int n_z) final;
    };
}