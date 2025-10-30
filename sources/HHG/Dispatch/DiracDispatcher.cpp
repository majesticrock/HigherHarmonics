#include "DiracDispatcher.hpp"
#include "../Laser/ContinuousLaser.hpp"
#include "../Laser/CosineLaser.hpp"
#include "../Laser/ExperimentalLaser.hpp"
#include "../Laser/QuenchedField.hpp"
#include "../Laser/PowerLawField.hpp"
#include "../Laser/DoubleLaser.hpp"

#include <iostream>
#include <chrono>

constexpr double target_kappa_error = 1e-3;
constexpr int n_kappa = 10;

HHG::Dispatch::DiracDispatcher::DiracDispatcher(mrock::utility::InputFileReader &input, int N, h_float t0_offset/* = h_float{} */)
    : Dispatcher(N, input.getDouble("diagonal_relaxation_time")),
    system(input.getDouble("T"), input.getDouble("E_F"), input.getDouble("v_F"), input.getDouble("band_width"), input.getDouble("photon_energy"), input.getDouble("diagonal_relaxation_time"))
{
    const h_float E0 = input.getDouble("field_amplitude");
    const h_float photon_energy = input.getDouble("photon_energy");
    const std::string laser_type = input.getString("laser_type");
    const int n_laser_cylces = input.getInt("n_laser_cycles");

    if (laser_type == "continuous") {
        laser = std::make_unique<Laser::ContinuousLaser>(photon_energy, E0, system.laser_model_ratio(photon_energy));
        time_config = {-n_laser_cylces * HHG::pi, n_laser_cylces * HHG::pi, N, 50};
    }
    else if (laser_type == "cosine") {
        laser = std::make_unique<Laser::CosineLaser>(photon_energy, E0, system.laser_model_ratio(photon_energy), n_laser_cylces, pi * t0_offset);
        time_config = {laser->t_begin, laser->t_end, N, 50};
    }
    else if (laser_type == "exp") {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, 
            system.laser_model_ratio(photon_energy * Laser::exp_photon_energy), t0_offset);
        time_config = {laser->t_begin, laser->t_end, N, 50};
    }
    else if (laser_type == "expA") {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, 
            system.laser_model_ratio(photon_energy * Laser::exp_photon_energy), t0_offset, Laser::ExperimentalLaser::Active::A);
        time_config = {laser->t_begin, laser->t_end, N, 50};
    }
    else if (laser_type == "expB") {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, 
            system.laser_model_ratio(photon_energy * Laser::exp_photon_energy), t0_offset, Laser::ExperimentalLaser::Active::B);
        time_config = {laser->t_begin, laser->t_end, N, 50};
    }
    else if (laser_type == "quench") {
        laser = std::make_unique<Laser::QuenchedField>(photon_energy, E0, system.laser_model_ratio(photon_energy), t0_offset);
        time_config = {laser->t_begin, laser->t_end, N, 50};
    }
    else if (laser_type.substr(0, powerlaw.size()) == powerlaw) {
        laser = std::make_unique<Laser::PowerLawField>(photon_energy, E0, system.laser_model_ratio(photon_energy), t0_offset, std::stod(laser_type.substr(powerlaw.size())));
        time_config = {laser->t_begin, laser->t_end, N, 50};
    }
    else {
        throw std::invalid_argument("Laser type '" + laser_type + "' is not recognized!");
    }
}

void HHG::Dispatch::DiracDispatcher::compute(int rank, int n_ranks, int n_z)
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;

    if (diagonal_relaxation_time > 0) {
        current_density_time = system.compute_current_density_diagonal_relaxation(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
    }
    else {
        current_density_time = system.compute_current_density(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

void HHG::Dispatch::DiracDispatcher::debug(int n_z)
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing the debug data sets..." << std::endl;

    if (diagonal_relaxation_time > 0) {
        time_evolutions = system.compute_current_density_diagonal_relaxation_debug(laser.get(), time_config, n_z, n_kappa, target_kappa_error);
    }
    else {
        time_evolutions = system.compute_current_density_debug(laser.get(), time_config, n_z, n_kappa, target_kappa_error);
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}
