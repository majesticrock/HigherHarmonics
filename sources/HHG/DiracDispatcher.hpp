#pragma once
#include "Dispatcher.hpp"
#include <mrock/utility/InputFileReader.hpp>
#include "DiracSystem.hpp"

namespace HHG {
    struct DiracDispatcher : public Dispatcher {
        DiracSystem system;

        DiracDispatcher(mrock::utility::InputFileReader& input, int N, h_float t0_offset = h_float{});

        void compute(int rank, int n_ranks, int n_z) final;
        void debug(int n_z) final;
    };
}