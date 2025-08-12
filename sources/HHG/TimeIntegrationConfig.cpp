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

    std::ostream& operator<<(std::ostream& os, TimeIntegrationConfig const& tconfig)
    {
        os << "TimeIntegrationConfig object:\nt_begin=" << tconfig.t_begin << "   t_end=" << tconfig.t_end 
            << "\nn_measurements=" << tconfig.n_measurements << "   n_subdivisions=" << tconfig.n_subdivisions
            << "\n";
        return os;
    }
}
