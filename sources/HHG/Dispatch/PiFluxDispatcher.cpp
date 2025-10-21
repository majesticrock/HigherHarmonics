#include "PiFluxDispatcher.hpp"
#include <iostream>
#include "../Laser/ContinuousLaser.hpp"
#include "../Laser/CosineLaser.hpp"
#include "../Laser/ExperimentalLaser.hpp"
#include "../Laser/QuenchedField.hpp"
#include "../Laser/PowerLawField.hpp"
#include "../Laser/DoubleCosine.hpp"

#include <chrono>

HHG::Dispatch::PiFluxDispatcher::PiFluxDispatcher(
    mrock::utility::InputFileReader &input, 
    int N, 
    h_float t0_offset /* = h_float{} */
)
    : Dispatcher(N, input.getDouble("diagonal_relaxation_time")),
      mod_system(input.getString("system_type") == "ModifiedPiFlux"),
      system([&]() -> std::variant<Systems::PiFlux, Systems::ModifiedPiFlux> {
          if (mod_system) {
              return Systems::ModifiedPiFlux(
                  input.getDouble("T"), 
                  input.getDouble("E_F"), 
                  input.getDouble("v_F"), 
                  input.getDouble("band_width"), 
                  Dispatcher::get_photon_energy(input), 
                  input.getDouble("diagonal_relaxation_time"),
                  input.getDouble("offdiagonal_relaxation_time") > h_float{} 
                      ? input.getDouble("offdiagonal_relaxation_time") 
                      : input.getDouble("diagonal_relaxation_time")
              );
          } else {
              return Systems::PiFlux(
                  input.getDouble("T"), 
                  input.getDouble("E_F"), 
                  input.getDouble("v_F"), 
                  input.getDouble("band_width"), 
                  Dispatcher::get_photon_energy(input), 
                  input.getDouble("diagonal_relaxation_time"),
                  input.getDouble("offdiagonal_relaxation_time") > h_float{} 
                      ? input.getDouble("offdiagonal_relaxation_time") 
                      : input.getDouble("diagonal_relaxation_time")
              );
          }
      }()),
      photon_energy(Dispatcher::get_photon_energy(input))
{
    const h_float E0 = input.getDouble("field_amplitude");
    const h_float photon_energy = input.getDouble("photon_energy");
    const std::string laser_type = input.getString("laser_type");
    const int n_laser_cycles = input.getInt("n_laser_cycles");

    const h_float laser_model_ratio = std::visit([](auto& sys) { return sys.laser_model_ratio(); }, system);
    const bool reduced_duration = input.getString("occupations") != "no" || input.getString("split_current") != "no";

    if (laser_type == continuous) {
        laser = std::make_unique<Laser::ContinuousLaser>(photon_energy, E0, laser_model_ratio);
        time_config = {-n_laser_cycles * HHG::pi, n_laser_cycles * HHG::pi, N, 50};
    }
    else if (laser_type == cosine) {
        laser = std::make_unique<Laser::CosineLaser>(photon_energy, E0, laser_model_ratio, n_laser_cycles, pi * t0_offset);
        // continue time evolution for 1 cycle so that the relaxation can set in
        time_config = {laser->t_begin, laser->t_end + (2. * pi), N, 50};
    }
    else if (laser_type == exp) {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, laser_model_ratio, t0_offset);
        time_config = {laser->t_begin, reduced_duration ? 0.75 * laser->t_end : laser->t_end, N, 50};
    }
    else if (laser_type == expA) {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, laser_model_ratio, t0_offset, Laser::ExperimentalLaser::Active::A);
        time_config = {laser->t_begin, reduced_duration ? 0.75 * laser->t_end : laser->t_end, N, 50};
    }
    else if (laser_type == expB) {
        laser = std::make_unique<Laser::ExperimentalLaser>(photon_energy, E0, laser_model_ratio, t0_offset, Laser::ExperimentalLaser::Active::B);
        time_config = {laser->t_begin, reduced_duration ? 0.75 * laser->t_end : laser->t_end, N, 50};
    }
    else if (laser_type == quench) {
        laser = std::make_unique<Laser::QuenchedField>(photon_energy, E0, laser_model_ratio, t0_offset);
        time_config = {laser->t_begin, laser->t_end, N, 50};
    }
    else if (laser_type.substr(0, powerlaw.size()) == powerlaw) {
        laser = std::make_unique<Laser::PowerLawField>(photon_energy, E0, laser_model_ratio, t0_offset, std::stod(laser_type.substr(powerlaw.size())));
        time_config = {laser->t_begin, laser->t_end, N, 50};
    }
    else if (laser_type == dcos) {
        laser = std::make_unique<Laser::DoubleCosine>(photon_energy, E0, laser_model_ratio, n_laser_cycles, t0_offset);
        time_config = {laser->t_begin, reduced_duration ? 0.75 * laser->t_end : laser->t_end, N, 50};
    }
    else if (laser_type == dcosA) {
        laser = std::make_unique<Laser::DoubleCosine>(photon_energy, E0, laser_model_ratio, n_laser_cycles, t0_offset, Laser::DoubleCosine::Active::A);
        time_config = {laser->t_begin, reduced_duration ? 0.75 * laser->t_end : laser->t_end, N, 50};
    }
    else if (laser_type == dcosB) {
        laser = std::make_unique<Laser::DoubleCosine>(photon_energy, E0, laser_model_ratio, n_laser_cycles, t0_offset, Laser::DoubleCosine::Active::B);
        time_config = {laser->t_begin, reduced_duration ? 0.75 * laser->t_end : laser->t_end, N, 50};
    }
    else {
        throw std::invalid_argument("Laser type '" + laser_type + "' is not recognized!");
    }

    std::cout << "Model parameters:" << std::endl;
    std::cout << "lattice constant = " << std::visit(
            [&](auto& sys){ return sys.get_property_in_SI_units("d", laser->photon_energy); },
            system
        ) << std::endl;
    std::cout << "Hopping element = " << std::visit(
            [&](auto& sys){ return sys.get_property_in_SI_units("t", laser->photon_energy); },
            system
        ) << std::endl;
}

void HHG::Dispatch::PiFluxDispatcher::compute(int rank, int n_ranks, int n_z)
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;

    current_density_time = std::visit(
        [&](auto& sys){ 
            return sys.compute_current_density(laser.get(), time_config, rank, n_ranks, n_z); 
        },
        system
    );

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

void HHG::Dispatch::PiFluxDispatcher::debug(int n_z)
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing the debug data sets..." << std::endl;

    if (auto pi_flux_ptr = std::get_if<Systems::PiFlux>(&system)) {
        time_evolutions = pi_flux_ptr->compute_current_density_debug(laser.get(), time_config, n_z);
    } else {
        throw std::runtime_error("compute_current_density_debug only supported for PiFlux system");
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

nlohmann::json HHG::Dispatch::PiFluxDispatcher::special_information() const
{
    return { 
        { "lattice_constant", std::visit(
            [&](auto& sys){ return sys.get_property_in_SI_units("d", photon_energy); },
            system
        )},
        { "hopping_element", std::visit(
            [&](auto& sys){ return sys.get_property_in_SI_units("t", photon_energy); },
            system
        )} 
    };
}

std::vector<HHG::OccupationContainer> HHG::Dispatch::PiFluxDispatcher::track_occupation_numbers(int N) const
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing occupation numbers..." << std::endl;

    std::vector<OccupationContainer> occupations = std::visit(
        [&](auto& sys){ 
            return sys.compute_occupation_numbers(laser.get(), time_config, N); 
        },
        system
    );

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    return occupations;
}

std::array<std::vector<HHG::h_float>, 2> HHG::Dispatch::PiFluxDispatcher::compute_split_current(int N) const
{
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::cout << "Computing split current..." << std::endl;

    if (auto pi_flux_ptr = std::get_if<Systems::PiFlux>(&system)) {
        auto result = pi_flux_ptr->current_per_energy(laser.get(), time_config, N); 

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	    std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

        return result;
    } 
    else {
        throw std::runtime_error("compute_current_density_debug only supported for PiFlux system");
    }
}