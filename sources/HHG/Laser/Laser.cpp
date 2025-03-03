#include "ContinuousLaser.hpp"

namespace HHG::Laser {
     /** 
     * converts e E_0 / (hbar omega_L) to 1 / pm
     * if E_0 is given in MV / cm and (hbar omega_L) in meV
     */
    constexpr h_float field_conversion = 1e-1; 

    Laser::Laser(h_float photon_energy, h_float E_0)
        : momentum_amplitude{field_conversion * E_0 / (photon_energy)} 
    {}

    Laser::Laser(h_float photon_energy, h_float E_0, h_float t_begin, h_float t_end)
        : momentum_amplitude{field_conversion * E_0 / (photon_energy)}, t_begin{t_begin}, t_end{t_end} 
    {}
}