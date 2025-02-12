#include "ContinuousLaser.hpp"

namespace hhg {
    Laser::Laser(h_float p_frequency, h_float p_momentum_amplitude)
        : frequency{p_frequency}, momentum_amplitude{p_momentum_amplitude} 
    {}
}