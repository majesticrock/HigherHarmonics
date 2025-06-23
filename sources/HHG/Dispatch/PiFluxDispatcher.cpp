#include "PiFluxDispatcher.hpp"
#include <iostream>
#include "../Laser/ContinuousLaser.hpp"
#include "../Laser/CosineLaser.hpp"
#include "../Laser/ExperimentalLaser.hpp"
#include "../Laser/QuenchedField.hpp"

#include <chrono>

HHG::Dispatch::PiFluxDispatcher::PiFluxDispatcher(mrock::utility::InputFileReader &input, int N, h_float t0_offset/* = h_float{} */)
    : Dispatcher(N, input.getDouble("diagonal_relaxation_time")),
    system(input.getDouble("T"), 
        input.getDouble("E_F"), 
        input.getDouble("v_F"), 
        input.getDouble("band_width"), 
        Dispatcher::get_photon_energy(input), 
        input.getDouble("diagonal_relaxation_time"),
        input.getDouble("offdiagonal_relaxation_time") > h_float{} ? input.getDouble("offdiagonal_relaxation_time") : input.getDouble("diagonal_relaxation_time"))
{
    const h_float E0 = input.getDouble("field_amplitude");
    const h_float photon_energy = input.getDouble("photon_energy");
    const std::string laser_type = input.getString("laser_type");
    const int n_laser_cylces = input.getInt("n_laser_cycles");

    if (laser_type == "continuous") {
        laser = std::make_unique<Laser::ContinuousLaser>(photon_energy, E0, system.laser_model_ratio());
        time_config = {-n_laser_cylces * HHG::pi, n_laser_cylces * HHG::pi, N, 500};
    }
    else if (laser_type == "cosine") {
        laser = std::make_unique<Laser::CosineLaser>(photon_energy, E0, system.laser_model_ratio(), n_laser_cylces, pi * t0_offset);
        // continue time evolution for 1 cycle so that the relaxation can set in
        time_config = {laser->t_begin, laser->t_end + (2. * pi), N, 500};
    }
    else if (laser_type == "exp") {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, system.laser_model_ratio(), t0_offset);
        time_config = {laser->t_begin, laser->t_end, N, 500};
    }
    else if (laser_type == "expA") {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, system.laser_model_ratio(), t0_offset, Laser::ExperimentalLaser::Active::A);
        time_config = {laser->t_begin, laser->t_end, N, 500};
    }
    else if (laser_type == "expB") {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, system.laser_model_ratio(), t0_offset, Laser::ExperimentalLaser::Active::B);
        time_config = {laser->t_begin, laser->t_end, N, 500};
    }
    else if (laser_type == "quench") {
        laser = std::make_unique<Laser::QuenchedField>(photon_energy, E0, system.laser_model_ratio(), t0_offset);
        time_config = {laser->t_begin, laser->t_end, N, 500};
    }
    else {
        throw std::invalid_argument("Laser type '" + laser_type + "' is not recognized!");
    }
}

void HHG::Dispatch::PiFluxDispatcher::compute(int rank, int n_ranks, int n_z)
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;

    current_density_time = system.compute_current_density(laser.get(), time_config, rank, n_ranks, n_z);

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

void HHG::Dispatch::PiFluxDispatcher::debug(int n_z)
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing the debug data sets..." << std::endl;

    time_evolutions = system.compute_current_density_debug(laser.get(), time_config, n_z);

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}
