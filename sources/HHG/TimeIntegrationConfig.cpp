#include "TimeIntegrationConfig.hpp"

namespace HHG {
    h_float TimeIntegrationConfig::dt() const
    {
        return measure_every() / n_subdivisions;
    }

    h_float TimeIntegrationConfig::measure_every() const
    {
        return (t_end - t_begin) / n_measurements;
    }
}
