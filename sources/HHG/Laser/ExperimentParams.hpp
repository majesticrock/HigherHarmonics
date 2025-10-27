#pragma once

namespace HHG::Laser {
    // Experimental data in ps [average temporal spacing of measurements]
    constexpr double exp_dt{ 0.03318960199004975 };
    constexpr int N_extra = 16;
    constexpr int N_experiment = 201;
    constexpr double laser_end{ 6.67111 + 2 * (N_extra - 1) * exp_dt }; ///< in ps [measured data ends at the first number, the second summand is the buffer for a smooth approach to 0]
    constexpr double exp_photon_energy{ 5.889401182228545 }; ///< in meV [obtained by FFT of the measured electric field]
    constexpr double unified_t_max{ 1.5 * laser_end }; ///< in ps; I just assume that I'll never use t0 > 0.5 * laser_end
}