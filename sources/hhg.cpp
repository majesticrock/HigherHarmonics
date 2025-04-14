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

constexpr double target_kappa_error = 5e-4;
constexpr int n_kappa = 10;
constexpr int zero_padding = 8;

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
    const int n_laser_cylces = input.getInt("n_laser_cycles");
    const int n_z = input.getInt("n_z");
    const h_float decay_time = input.getDouble("decay_time"); // in fs
    const std::string debug_data = input.getString("debug_data"); // yes/no/only

    constexpr int measurements_per_cycle = 1 << 14; // 2^14 is the mininum value to achieve good precision for realistic parameters
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

    DiracSystem system(temperature, E_F, v_F, band_width, photon_energy, decay_time);
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
        + "/decay_time=" + improved_string(decay_time > 0 ? decay_time : -1)
        + "/";
    const std::string output_dir = BASE_DATA_DIR + data_subdir;
    std::filesystem::create_directories(output_dir);

    /**
     * Starting calculations
     */

    high_resolution_clock::time_point begin = high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;

    std::array<std::vector<h_float>, DiracSystem::n_debug_points> time_evolutions;
    std::vector<h_float> current_density_time;
#ifndef NO_MPI
    std::vector<h_float> current_density_time_local;

    if (decay_time > 0) {
        if (rank == 0 && debug_data != "no")
            time_evolutions = system.compute_current_density_decay_debug(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
        if (debug_data == "only")
            current_density_time_local.resize(N + 1);
        else
            current_density_time_local = system.compute_current_density_decay(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
    }
    else {
        if (rank == 0 && debug_data != "no")
            time_evolutions = system.compute_current_density_debug(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
        if (debug_data == "only")
            current_density_time_local.resize(N + 1);
        else
            current_density_time_local = system.compute_current_density(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error); 
    }

    current_density_time.resize(current_density_time_local.size());
    MPI_Reduce(current_density_time_local.data(), current_density_time.data(), current_density_time_local.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    if (decay_time > 0) {
        if (debug_data != "only")
            current_density_time = system.compute_current_density_decay(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
        if (debug_data != "no")
            time_evolutions = system.compute_current_density_decay_debug(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
    }
    else {
        if (debug_data != "only")
            current_density_time = system.compute_current_density(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
        if (debug_data != "no")
            time_evolutions = system.compute_current_density_debug(laser.get(), time_config, rank, n_ranks, n_z, n_kappa, target_kappa_error);
    }
#endif

    high_resolution_clock::time_point end = high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    if (rank != 0) return EXIT;
    /////////////////////////////////////////////////////////////////////

    begin = high_resolution_clock::now();
    std::cout << "Computing the Fourier transform..." << std::endl;

    std::vector<h_float> current_density_frequency_real(N + 1);
    std::vector<h_float> current_density_frequency_imag(N + 1);

    std::array<std::vector<h_float>, DiracSystem::n_debug_points> debug_fft_real;
    std::array<std::vector<h_float>, DiracSystem::n_debug_points> debug_fft_imag;
    debug_fft_real.fill(std::vector<h_float>(N + 1));
    debug_fft_imag.fill(std::vector<h_float>(N + 1));

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
        HHG::Fourier::TrapezoidalFFT integrator(time_config);

        // 0 padding to increase frequency resolution. Factor >= 4 is recommended by numerical recipes
        current_density_time.resize(zero_padding * (N + 1));
        current_density_frequency_real.resize(zero_padding * (N + 1));
        current_density_frequency_imag.resize(zero_padding * (N + 1));

        integrator.compute(current_density_time, current_density_frequency_real, current_density_frequency_imag);
        frequencies = integrator.frequencies;

        if (debug_data != "no") {
            for(int i = 0; i < DiracSystem::n_debug_points; ++i) {
                integrator.compute(time_evolutions[i], debug_fft_real[i], debug_fft_imag[i]);
            }
        }
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

    // We do not need to output the zero padding
    std::vector<h_float> current_density_time_output(current_density_time.begin(), current_density_time.begin() + N + 1);
    nlohmann::json data_json {
        { "time", 				                mrock::utility::time_stamp() },
        { "laser_function",                     laser_function },
        { "N",                                  N },
        { "t_begin",                            time_config.t_begin },
        { "t_end",                              time_config.t_end },
        { "n_laser_cycles",                     n_laser_cylces },
        { "n_measurements",                     time_config.n_measurements + int(laser_type != "continuous") },
        { "current_density_time",               current_density_time_output },
        { "current_density_frequency_real",     current_density_frequency_real },
        { "current_density_frequency_imag",     current_density_frequency_imag },
        { "T",                                  temperature },
        { "E_F",                                E_F },
        { "v_F",                                v_F },
        { "band_width",                         band_width },
        { "field_amplitude",                    E0 },
        { "photon_energy",                      photon_energy },
        { "laser_type",                         laser_type },
        { "decay_time",                         decay_time },
        { "frequencies",                        frequencies },
        { "zero_padding",                       zero_padding }
    };
    std::cout << "Saving data to " << output_dir << "/current_density.json.gz" << std::endl;
    mrock::utility::saveString(data_json.dump(4), output_dir + "current_density.json.gz");
    if (debug_data == "no") return EXIT;

    // Debug output
    nlohmann::json debug_json = {
        {"time", mrock::utility::time_stamp()},
        {"frequencies", frequencies},
        {"time_evolutions", time_evolutions},
        {"debug_fft_real", debug_fft_real},
        {"debug_fft_imag", debug_fft_imag}
    };
    mrock::utility::saveString(debug_json.dump(4), output_dir + "time_evolution.json.gz");

    return EXIT;
}