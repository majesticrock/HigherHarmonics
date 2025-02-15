#include "ContinuousLaser.hpp"

namespace HHG {
    Laser::Laser(h_float photon_energy, h_float E_0)
        : momentum_amplitude{field_conversion * E_0 / (photon_energy)} 
    {}
}