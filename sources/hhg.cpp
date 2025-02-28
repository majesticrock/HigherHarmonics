#include <iostream>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <chrono>

#include <nlohmann/json.hpp>
#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/InputFileReader.hpp>
#include <mrock/utility/better_to_string.hpp>
#include <mrock/utility/Numerics/Integration/AdaptiveTrapezoidalRule.hpp>
#include <mrock/utility/Numerics/ErrorFunctors.hpp>

#include "HHG/DiracSystem.hpp"
#include "HHG/ContinuousLaser.hpp"
#include "HHG/FFT.hpp"
#include "HHG/WelchWindow.hpp"

constexpr int n_kappa = 500;
constexpr int n_z = 200;

// k_z * kappa / |k| is found analytically (compare the HHG document)
inline HHG::h_float integration_weight(HHG::h_float k_z, HHG::h_float kappa) {
    return k_z * kappa / HHG::norm(k_z, kappa);
}

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

int main(int argc, char** argv) {
    using namespace HHG;
    using std::chrono::high_resolution_clock;

    if (argc < 2) {
		std::cerr << "Invalid number of arguments: Use mpirun -n <threads> <path_to_executable> <configfile>" << std::endl;
		return -1;
	}
    mrock::utility::InputFileReader input(argv[1]);

    h_float temperature = input.getDouble("T");
    h_float E_F = input.getDouble("E_F");
    h_float v_F = input.getDouble("v_F");
    h_float band_width = input.getDouble("band_width");
    h_float E0 = input.getDouble("field_amplitude");
    h_float photon_energy = input.getDouble("photon_energy");

    constexpr int n_laser_cylces = 1 << 6; // Increase this to increase frequency resolution Delta omega
    constexpr int measurements_per_cycle = 1 << 6; // Decrease this to reduce the cost of the FFT
    constexpr int N = n_laser_cylces * measurements_per_cycle;

    TimeIntegrationConfig time_config = {-n_laser_cylces * HHG::pi, n_laser_cylces * HHG::pi, N, 500};
    DiracSystem system(temperature, E_F, v_F, band_width, photon_energy);
    std::unique_ptr<Laser> laser = std::make_unique<ContinuousLaser>(photon_energy, E0);

    nd_vector rhos_buffer = nd_vector::Zero(N);
    std::vector<h_float> current_density_time(N, h_float{});

    const int pick_time = 987;
    constexpr int pick_z = n_z / 5;

    nd_vector pick_current_density_z = nd_vector::Zero(n_z + 1);
    std::map<h_float, h_float> pick_current_density_kappa;
    std::map<h_float, h_float> pick_current_density_kappa_minus;

    high_resolution_clock::time_point begin = high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;

    const auto delta_z = 2 * system.z_integration_upper_limit() / n_z;

    mrock::utility::Numerics::Integration::adapative_trapezoidal_rule<h_float, mrock::utility::Numerics::Integration::adapative_trapezoidal_rule_print_policy{false, false}> integrator;

#pragma omp parallel for firstprivate(rhos_buffer) reduction(vec_plus:current_density_time)
    for (int z = 1; z < n_z; ++z) { // f(|z| = z_max) = 0
        std::cout << "z=" << z << std::endl;
        const auto k_z = (z - n_z / 2) * delta_z;
        auto kappa_integrand = [&](h_float kappa) -> const nd_vector& {
            if (is_zero(kappa)) {
                rhos_buffer.setZero();
            }
            else {
                system.time_evolution_sigma(rhos_buffer, laser.get(), k_z, kappa, time_config);
                rhos_buffer *= integration_weight(k_z, kappa);
            } 

            if (z == n_z - pick_z) {
                pick_current_density_kappa[kappa] = rhos_buffer[pick_time];
            }
            if (z == pick_z) {
                pick_current_density_kappa_minus[kappa] = rhos_buffer[pick_time];
            }
            return rhos_buffer;
        };

        auto kappa_result = integrator.integrate(kappa_integrand, h_float{}, system.kappa_integration_upper_limit(k_z), 
            n_kappa, 1e-3, mrock::utility::Numerics::vector_elementwise_error<nd_vector, h_float, false>(), nd_vector::Zero(N));
        pick_current_density_z[z] = kappa_result[pick_time];

        std::transform(current_density_time.begin(), current_density_time.end(), kappa_result.begin(), current_density_time.begin(), std::plus<>());
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    begin = high_resolution_clock::now();
    std::cout << "Computing the FFT..." << std::endl;

    FFT fft(N);

    WelchWindow window(N);
    for (int i = 0; i < N; ++i) {
        current_density_time[i] *= delta_z * window[i];
    }
    std::vector<h_float> current_density_frequency_real(N);
    std::vector<h_float> current_density_frequency_imag(N);
    fft.compute(current_density_time, current_density_frequency_real, current_density_frequency_imag);

    end = high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    std::vector<HHG::h_float> laser_function(N);
    for (int i = 0; i < N; i++)
    {
        laser_function[i] = laser->laser_function(time_config.t_begin + i * time_config.measure_every());
    }

    std::vector<h_float> k_zs(n_z + 1);
    for (int i = 0; i <= n_z; ++i) {
        k_zs[i] = (i - n_z / 2) * delta_z;
    }

    nlohmann::json data_json {
        { "time", 				                mrock::utility::time_stamp() },
        { "laser_function",                     laser_function },
        { "N",                                  N },
        { "t_begin",                            time_config.t_begin },
        { "t_end",                              time_config.t_end },
        { "n_laser_cycles",                     n_laser_cylces },
        { "n_measurements",                     time_config.n_measurements },
        { "current_density_time",               current_density_time },
        { "current_density_frequency_real",     current_density_frequency_real },
        { "current_density_frequency_imag",     current_density_frequency_imag },
        { "T",                                  temperature },
        { "E_F",                                E_F },
        { "v_F",                                v_F },
        { "band_width",                         band_width },
        { "field_amplitude",                    E0 },
        { "photon_energy",                      photon_energy },
        { "k_zs",                               k_zs },
        { "pick_current_density_z",             pick_current_density_z },
        { "pick_current_density_kappa",         pick_current_density_kappa },
        { "pick_current_density_kappa_minus",   pick_current_density_kappa_minus },
    };

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
        + "/T=" + improved_string(temperature)
        + "/E_F=" + improved_string(E_F)
        + "/v_F=" + improved_string(v_F)
        + "/band_width=" + improved_string(band_width)
        + "/field_amplitude=" + improved_string(E0)
        + "/photon_energy=" + improved_string(photon_energy) 
        + "/";
    const std::string output_dir = BASE_DATA_DIR + data_subdir;

    std::filesystem::create_directories(output_dir);
    mrock::utility::saveString(data_json.dump(4), output_dir + "current_density.json.gz");

    return 0;
}