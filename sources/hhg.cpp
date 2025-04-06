#include <iostream>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <chrono>

#ifndef NO_MPI
#include <mpi.h>
#define EXIT MPI_Finalize()
#else
#define EXIT 0
#endif

#include <nlohmann/json.hpp>
#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/InputFileReader.hpp>
#include <mrock/utility/better_to_string.hpp>

#include "HHG/DiracSystem.hpp"
#include "HHG/Laser/ContinuousLaser.hpp"
#include "HHG/Laser/CosineLaser.hpp"
#include "HHG/Fourier/FFT.hpp"
#include "HHG/Fourier/WelchWindow.hpp"
#include "HHG/Fourier/FourierIntegral.hpp"
#include "HHG/Fourier/TrapezoidalFFT.hpp"

constexpr double target_kappa_error = 5e-3;
constexpr int n_kappa = 10;

int main(int argc, char** argv) {
    using namespace HHG;
    using std::chrono::high_resolution_clock;

    if (argc < 2) {
		std::cerr << "Invalid number of arguments: Use mpirun -n <threads> <path_to_executable> <configfile>" << std::endl;
		return -1;
	}

#ifndef NO_MPI
	MPI_Init(&argc, &argv);
	int rank, n_ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
#else
	int rank = 0;
	int n_ranks = 1;
#endif

    mrock::utility::InputFileReader input(argv[1]);

    /**
     * Loading configurations
     */
    const h_float temperature = input.getDouble("T");
    const h_float E_F = input.getDouble("E_F");
    const h_float v_F = input.getDouble("v_F");
    const h_float band_width = input.getDouble("band_width");
    const h_float E0 = input.getDouble("field_amplitude");
    const h_float photon_energy = input.getDouble("photon_energy");
    const std::string laser_type = input.getString("laser_type");
    const int n_laser_cylces = input.getInt("n_laser_cycles"); // Increase this to increase frequency resolution Delta omega
    const int n_z = input.getInt("n_z");

    constexpr int measurements_per_cycle = 1 << 8; // Decrease this to reduce the cost of the FFT
    const int N = n_laser_cylces * measurements_per_cycle;

    std::unique_ptr<Laser::Laser> laser;
    TimeIntegrationConfig time_config;
    if (laser_type == "continuous") {
        laser = std::make_unique<Laser::ContinuousLaser>(photon_energy, E0, v_F);
        time_config = {-n_laser_cylces * HHG::pi, n_laser_cylces * HHG::pi, N, 500};
    }
    else if (laser_type == "cosine") {
        laser = std::make_unique<Laser::CosineLaser>(photon_energy, E0, v_F, n_laser_cylces);
        time_config = {laser->t_begin, laser->t_end, N, 500};
    }
    else {
        std::cerr << "Laser type '" << laser_type << "' is not recognized!" << std::endl;
        return 1;
    }

    DiracSystem system(temperature, E_F, v_F, band_width, photon_energy);

    //std::cout << laser->momentum_amplitude / (photon_energy * 1e-3) << std::endl;

    /**
     * Creating output dirs
     */
    auto improved_string = [](h_float number) -> std::string {
        if (std::floor(number) == number) {
            // If the number is a whole number, format it with one decimal place
            std::ostringstream out;
            out.precision(1);
            out << std::fixed << number;
            return out.str();
        }
        else {
            std::string str = mrock::utility::better_to_string(number, std::chars_format::fixed);
            // Remove trailing zeroes
            str.erase(str.find_last_not_of('0') + 1, std::string::npos);
            str.erase(str.find_last_not_of('.') + 1, std::string::npos);
            return str;
        }
    };

    const std::string BASE_DATA_DIR = "../../data/HHG/";
    const std::string data_subdir = input.getString("data_dir")
        + "/" + laser_type + "_laser" +
        + "/T=" + improved_string(temperature)
        + "/E_F=" + improved_string(E_F)
        + "/v_F=" + improved_string(v_F)
        + "/band_width=" + improved_string(band_width)
        + "/field_amplitude=" + improved_string(E0)
        + "/photon_energy=" + improved_string(photon_energy) 
        + "/";
    const std::string output_dir = BASE_DATA_DIR + data_subdir;
    std::filesystem::create_directories(output_dir);

    /**
     * Starting calculations
     */

    high_resolution_clock::time_point begin = high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;

#ifndef NO_MPI
    std::vector<h_float> current_density_time_local = Dirac::compute_current_density(system, laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error, output_dir);
    std::vector<h_float> current_density_time(current_density_time_local.size());
    MPI_Reduce(current_density_time_local.data(), current_density_time.data(), current_density_time_local.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    std::vector<h_float> current_density_time = Dirac::compute_current_density(system, laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error, output_dir);
#endif

    high_resolution_clock::time_point end = high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    if (rank != 0) return EXIT;
    /////////////////////////////////////////////////////////////////////

    begin = high_resolution_clock::now();
    std::cout << "Computing the Fourier transform..." << std::endl;

    std::vector<h_float> current_density_frequency_real(N + 1);
    std::vector<h_float> current_density_frequency_imag(N + 1);

    std::vector<h_float> frequencies;
    if (laser_type == "continuous") {
        Fourier::WelchWindow window(N);
        for (int i = 0; i < N; ++i) {
            current_density_time[i] *= window[i];
        }
        Fourier::FFT fft(N);
        fft.compute(current_density_time, current_density_frequency_real, current_density_frequency_imag);

        frequencies.resize(N / 2 + 1);
        for (int i = 0; i < frequencies.size(); ++i) {
            frequencies[i] = i / time_config.measure_every();
        }
    }
    else {
        //HHG::Fourier::FourierIntegral integrator(time_config);
        HHG::Fourier::TrapezoidalFFT integrator(time_config);
        integrator.compute(current_density_time, current_density_frequency_real, current_density_frequency_imag);
        frequencies = integrator.frequencies;
    }

    end = high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    /**
     * Saving data
     */
    std::vector<HHG::h_float> laser_function(N + int(laser_type != "continuous"));
    for (int i = 0; i < laser_function.size(); i++)
    {
        laser_function[i] = laser->laser_function(time_config.t_begin + i * time_config.measure_every());
    }

    nlohmann::json data_json {
        { "time", 				                mrock::utility::time_stamp() },
        { "laser_function",                     laser_function },
        { "N",                                  N },
        { "t_begin",                            time_config.t_begin },
        { "t_end",                              time_config.t_end },
        { "n_laser_cycles",                     n_laser_cylces },
        { "n_measurements",                     time_config.n_measurements + int(laser_type != "continuous") },
        { "current_density_time",               current_density_time },
        { "current_density_frequency_real",     current_density_frequency_real },
        { "current_density_frequency_imag",     current_density_frequency_imag },
        { "T",                                  temperature },
        { "E_F",                                E_F },
        { "v_F",                                v_F },
        { "band_width",                         band_width },
        { "field_amplitude",                    E0 },
        { "photon_energy",                      photon_energy },
        { "laser_type",                         laser_type },
        { "frequencies",                        frequencies }
    };
    mrock::utility::saveString(data_json.dump(4), output_dir + "current_density.json.gz");

    return EXIT;
}