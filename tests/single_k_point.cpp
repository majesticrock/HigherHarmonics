#include <iostream>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/InputFileReader.hpp>

#include "HHG/DiracSystem.hpp"
#include "HHG/ContinuousLaser.hpp"
#include "HHG/FFT.hpp"

#include "HHG/GreensFunction.hpp"

//#define no_compute_greens_function

int main(int argc, char** argv)
{
    if (argc < 2) {
		std::cerr << "Invalid number of arguments: Use mpirun -n <threads> <path_to_executable> <configfile>" << std::endl;
		return -1;
	}
    mrock::utility::InputFileReader input(argv[1]);

    HHG::h_float E0 = input.getDouble("field_amplitude");
    HHG::h_float E_F = input.getDouble("E_F");
    HHG::h_float v_F = input.getDouble("v_F");
    HHG::h_float photon_energy = input.getDouble("photon_energy");
    HHG::h_float band_width = input.getDouble("band_width");

    std::cout << "Starting computations for\n"
        << " E_0 = " << E0 << " MV / cm\n"
        << " E_F = " << E_F << " meV\n"
        << " v_F = " << v_F << " m/s\n"
        << " photon_energy = " << photon_energy << " meV\n"
        << " band_width = " << band_width << " * photon_energy\n"
        << std::endl;

    constexpr int n_laser_cylces = 1 << 6; // Increase this to increase frequency resolution Delta omega
    constexpr int measurements_per_cycle = 1 << 6; // Decrease this to reduce the cost of the FFT
    constexpr int N = n_laser_cylces * measurements_per_cycle;

    constexpr HHG::h_float k_z = 1.1;
    constexpr HHG::h_float kappa = 0.1;

    HHG::DiracSystem system(E_F, v_F, band_width, photon_energy);
    HHG::ContinuousLaser laser(photon_energy, E0);
    HHG::TimeIntegrationConfig time_config = {-n_laser_cylces * HHG::pi, n_laser_cylces * HHG::pi, N, 500};

    std::cout << system.info() << std::endl;

    std::vector<HHG::h_float> laser_function(N + 1);
    for (int i = 0; i <= N; i++)
    {
        laser_function[i] = laser.laser_function(time_config.t_begin + i * time_config.measure_every());
    }

    std::vector<HHG::h_float> alphas, betas;
    system.time_evolution(alphas, betas, &laser, k_z, kappa, time_config);

    HHG::FFT fft(N);
    std::vector<HHG::h_float> rho(N);
    for(int i = 0; i < N; ++i) {
        rho[i] = alphas[i] - betas[i];
    }
    std::vector<HHG::h_float> fourier_rho_real;
    std::vector<HHG::h_float> fourier_rho_imag;
    fft.compute(rho, fourier_rho_real, fourier_rho_imag);

#ifndef no_compute_greens_function
    // Greens function
    auto c_time_evolution = [&system](std::vector<HHG::h_complex>& c_alphas, std::vector<HHG::h_complex>& c_betas, HHG::Laser const * const laser, 
        HHG::h_float k_z, HHG::h_float kappa, const HHG::TimeIntegrationConfig& time_config) 
    {
        system.time_evolution_complex(c_alphas, c_betas, laser, k_z, kappa, time_config);
    };
    HHG::GreensFunction greens_function(c_time_evolution, N, measurements_per_cycle);
    greens_function.compute_alphas_betas(&laser, k_z, kappa, time_config);
    greens_function.compute_time_domain_greens_function(0, system.dispersion(k_z, kappa) * time_config.measure_every());
    greens_function.fourier_transform_greens_function();
#endif

    nlohmann::json data_json {
        { "time", 				    mrock::utility::time_stamp() },
        { "laser_function",         laser_function },
        { "N",                      N },
        { "alphas",                 alphas },
        { "betas",                  betas },
        { "t_begin",                time_config.t_begin },
        { "t_end",                  time_config.t_end },
        { "n_laser_cycles",         n_laser_cylces },
        { "n_measurements",         time_config.n_measurements },
        { "n_subdivisions",         time_config.n_subdivisions },
        { "fourier_rho_real",       fourier_rho_real },
        { "fourier_rho_imag",       fourier_rho_imag },
#ifndef no_compute_greens_function
        { "fourier_greens_real",    greens_function.get_fourier_greens_function_real() },
        { "fourier_greens_imag",    greens_function.get_fourier_greens_function_imag() },
        { "time_greens_real",       greens_function.get_time_domain_greens_function_real() },
        { "time_greens_imag",       greens_function.get_time_domain_greens_function_imag() },
#endif
    };

    std::filesystem::create_directories("../../data/HHG/test/");
    mrock::utility::saveString(data_json.dump(4), "../../data/HHG/test/test_data.json.gz");

    return 0;
}