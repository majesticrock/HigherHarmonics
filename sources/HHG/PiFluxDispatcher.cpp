#include "PiFluxDispatcher.hpp"
#include <iostream>
#include "Laser/ContinuousLaser.hpp"
#include "Laser/CosineLaser.hpp"

#include <chrono>
#ifndef NO_MPI
#include <mpi.h>
#endif

HHG::PiFluxDispatcher::PiFluxDispatcher(mrock::utility::InputFileReader &input, int N)
    : Dispatcher(N, input.getDouble("decay_time")),
    system(input.getDouble("T"), input.getDouble("E_F"), input.getDouble("v_F"), input.getDouble("band_width"), input.getDouble("photon_energy"), input.getDouble("decay_time"))
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
        laser = std::make_unique<Laser::CosineLaser>(photon_energy, E0, system.laser_model_ratio(), n_laser_cylces);
        time_config = {laser->t_begin, laser->t_end, N, 500};
    }
    else {
        throw std::invalid_argument("Laser type '" + laser_type + "' is not recognized!");
    }
}

void HHG::PiFluxDispatcher::compute(int rank, int n_ranks, int n_z)
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;

#ifndef NO_MPI
    std::vector<h_float> current_density_time_local;

    if (decay_time > 0) {
        std::cerr << "DECAY NOT YET IMPLEMENTED!" << std::endl;
        //current_density_time_local = system.compute_current_density_decay(laser.get(), time_config, rank, n_ranks, n_z);
    }
    else {
        current_density_time_local = system.compute_current_density(laser.get(), time_config, rank, n_ranks, n_z); 
    }

    current_density_time.resize(current_density_time_local.size());
    MPI_Reduce(current_density_time_local.data(), current_density_time.data(), current_density_time_local.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    if (decay_time > 0) {
        std::cerr << "DECAY NOT YET IMPLEMENTED!" << std::endl;
        //current_density_time = system.compute_current_density_decay(laser.get(), time_config, rank, n_ranks, n_z);
    }
    else {
        current_density_time = system.compute_current_density(laser.get(), time_config, rank, n_ranks, n_z);
    }
#endif

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

void HHG::PiFluxDispatcher::debug(int n_z)
{
    std::cerr << "DEBUG PI FLUX NOT YET IMPLEMENTED!" << std::endl;
}
