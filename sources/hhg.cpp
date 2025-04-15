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

#include "HHG/DiracDispatcher.hpp"
#include "HHG/PiFluxDispatcher.hpp"

#include "HHG/Fourier/FFT.hpp"
#include "HHG/Fourier/WelchWindow.hpp"
#include "HHG/Fourier/FourierIntegral.hpp"
#include "HHG/Fourier/TrapezoidalFFT.hpp"

#include <mrock/utility/info_to_json.hpp>
#include <mrock/info.h>
#include "../build_header/info.h"

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
    const h_float decay_time = input.getDouble("decay_time"); // in fs

    const std::string laser_type = input.getString("laser_type");
    const int n_laser_cylces = input.getInt("n_laser_cycles");
    const int n_z = input.getInt("n_z");
    
    const std::string system_type = input.getString("system_type");
    const std::string debug_data = input.getString("debug_data"); // yes/no/only

    constexpr int measurements_per_cycle = 1 << 14; // 2^14 is the mininum value to achieve good precision for realistic parameters
    const int N = n_laser_cylces * measurements_per_cycle;

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
    std::unique_ptr<Dispatcher> dispatcher;
    if (system_type == "Dirac") {
        dispatcher = std::make_unique<DiracDispatcher>(input, N + 1);
    }
    else if (system_type == "PiFlux") {
        dispatcher = std::make_unique<PiFluxDispatcher>(input, N + 1);
    }
    else {
        throw std::invalid_argument("System type '" + system_type + "' not recognized!");
    }

    if (debug_data != "only") dispatcher->compute(rank, n_ranks, n_z);
    if (debug_data != "no") dispatcher->debug(n_z);

    if (rank != 0) return EXIT;
    /////////////////////////////////////////////////////////////////////

    high_resolution_clock::time_point begin = high_resolution_clock::now();
    std::cout << "Computing the Fourier transform..." << std::endl;

    std::vector<h_float> current_density_frequency_real(N + 1);
    std::vector<h_float> current_density_frequency_imag(N + 1);

    std::array<std::vector<h_float>, n_debug_points> debug_fft_real;
    std::array<std::vector<h_float>, n_debug_points> debug_fft_imag;
    debug_fft_real.fill(std::vector<h_float>(N + 1));
    debug_fft_imag.fill(std::vector<h_float>(N + 1));

    std::vector<h_float> frequencies;
    if (laser_type == "continuous") {
        Fourier::WelchWindow window(N);
        for (int i = 0; i < N; ++i) {
            dispatcher->current_density_time[i] *= window[i];
        }
        Fourier::FFT fft(N);
        fft.compute(dispatcher->current_density_time, current_density_frequency_real, current_density_frequency_imag);

        frequencies.resize(N / 2 + 1);
        for (int i = 0; i < frequencies.size(); ++i) {
            frequencies[i] = i / dispatcher->time_config.measure_every();
        }
    }
    else {
        HHG::Fourier::TrapezoidalFFT integrator(dispatcher->time_config);

        // 0 padding to increase frequency resolution. Factor >= 4 is recommended by numerical recipes
        dispatcher->current_density_time.resize(zero_padding * (N + 1));
        current_density_frequency_real.resize(zero_padding * (N + 1));
        current_density_frequency_imag.resize(zero_padding * (N + 1));

        integrator.compute(dispatcher->current_density_time, current_density_frequency_real, current_density_frequency_imag);
        frequencies = integrator.frequencies;

        if (debug_data != "no") {
            for(int i = 0; i < n_debug_points; ++i) {
                integrator.compute(dispatcher->time_evolutions[i], debug_fft_real[i], debug_fft_imag[i]);
            }
        }
    }

    high_resolution_clock::time_point end = high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    /**
     * Saving data
     */
    std::vector<HHG::h_float> laser_function(N + int(laser_type != "continuous"));
    for (int i = 0; i < laser_function.size(); i++)
    {
        laser_function[i] = dispatcher->laser->laser_function(dispatcher->time_config.t_begin + i * dispatcher->time_config.measure_every());
    }

    // We do not need to output the zero padding
    std::vector<h_float> current_density_time_output(dispatcher->current_density_time.begin(), dispatcher->current_density_time.begin() + N + 1);
    nlohmann::json data_json {
        { "time", 				                mrock::utility::time_stamp() },
        { "laser_function",                     laser_function },
        { "N",                                  N },
        { "t_begin",                            dispatcher->time_config.t_begin },
        { "t_end",                              dispatcher->time_config.t_end },
        { "n_laser_cycles",                     n_laser_cylces },
        { "n_measurements",                     dispatcher->time_config.n_measurements + int(laser_type != "continuous") },
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

    // Generate metadata
	nlohmann::json info_json = mrock::utility::generate_json<HigherHarmonics::info>("hhg_");
	info_json.update(mrock::utility::generate_json<mrock::info>("mrock_"));
	mrock::utility::saveString(info_json.dump(4), output_dir + "metadata.json.gz");

    if (debug_data == "no") return EXIT;

    // Debug output
    nlohmann::json debug_json = {
        {"time", mrock::utility::time_stamp()},
        {"frequencies", frequencies},
        {"time_evolutions", dispatcher->time_evolutions},
        {"debug_fft_real", debug_fft_real},
        {"debug_fft_imag", debug_fft_imag}
    };
    mrock::utility::saveString(debug_json.dump(4), output_dir + "time_evolution.json.gz");

    return EXIT;
}