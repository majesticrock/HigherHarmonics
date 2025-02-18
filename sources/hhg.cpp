#include <iostream>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <chrono>

#include <nlohmann/json.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/InputFileReader.hpp>
#include <mrock/utility/better_to_string.hpp>

#include "HHG/DiracSystem.hpp"
#include "HHG/ContinuousLaser.hpp"
#include "HHG/FFT.hpp"

constexpr int n_kappa = 120;
constexpr int n_z = 120;
typedef boost::math::quadrature::gauss<HHG::h_float, n_kappa> kappa_integrator;
typedef boost::math::quadrature::gauss<HHG::h_float, n_z> z_integrator;

// k_z * kappa / |k| is found analytically (compare the HHG document)
inline HHG::h_float integration_weight(HHG::h_float k_z, HHG::h_float kappa) {
    return k_z * kappa / HHG::norm(k_z, kappa);
}

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

    std::vector<h_float> alpha_buffer(N);
    std::vector<h_float> beta_buffer(N);

    std::vector<h_float> alphas(N, h_float{});
    std::vector<h_float> betas(N, h_float{});


    high_resolution_clock::time_point begin = high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;
    for (int z = 0; z < n_z / 2; ++z) {
        const auto k_z = system.convert_to_z_integration(z_integrator::abscissa()[z]);
        const auto weight_z = z_integrator::weights()[z];

        for (int r = 0; r < n_kappa / 2; ++r) {
            // positive abcissae for kappa
            auto kappa = system.convert_to_kappa_integration(kappa_integrator::abscissa()[r], k_z);
            auto weight = kappa_integrator::weights()[r] * weight_z * integration_weight(k_z, kappa);
            
            system.time_evolution(alpha_buffer, beta_buffer, laser.get(), k_z, kappa, time_config);
            for (int i = 0; i < N; ++i) {
                alphas[i] += weight * alpha_buffer[i];
                betas[i]  += weight * beta_buffer[i];
            }
            system.time_evolution(alpha_buffer, beta_buffer, laser.get(), -k_z, kappa, time_config);
            for (int i = 0; i < N; ++i) { // The weight has the opposite sign for -k_z
                alphas[i] -= weight * alpha_buffer[i];
                betas[i]  -= weight * beta_buffer[i];
            }
            
            // negative abcissae for kappa
            kappa = system.convert_to_kappa_integration(-kappa_integrator::abscissa()[r], k_z);
            weight = kappa_integrator::weights()[r] * weight_z * integration_weight(k_z, kappa);
            
            system.time_evolution(alpha_buffer, beta_buffer, laser.get(), k_z, kappa, time_config);
            for (int i = 0; i < N; ++i) {
                alphas[i] += weight * alpha_buffer[i];
                betas[i]  += weight * beta_buffer[i];
            }
            system.time_evolution(alpha_buffer, beta_buffer, laser.get(), -k_z, kappa, time_config);
            for (int i = 0; i < N; ++i) { // The weight has the opposite sign for -k_z
                alphas[i] -= weight * alpha_buffer[i];
                betas[i]  -= weight * beta_buffer[i];
            }
        }
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    begin = high_resolution_clock::now();
    std::cout << "Computing the FFT..." << std::endl;

    FFT fft(N);
    std::vector<h_float> current_density_time = alphas;
    for (int i = 0; i < N; ++i) {
        current_density_time[i] -= betas[i];
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

    nlohmann::json data_json {
        { "time", 				                mrock::utility::time_stamp() },
        { "laser_function",                     laser_function },
        { "N",                                  N },
        { "alphas",                             alphas },
        { "betas",                              betas },
        { "t_begin",                            time_config.t_begin },
        { "t_end",                              time_config.t_end },
        { "n_laser_cycles",                     n_laser_cylces },
        { "n_measurements",                     time_config.n_measurements },
        { "current_density_frequency_real",     current_density_frequency_real },
        { "current_density_frequency_imag",     current_density_frequency_imag },
        { "T",                                  temperature },
        { "E_F",                                E_F },
        { "v_F",                                v_F },
        { "band_width",                         band_width },
        { "field_amplitude",                    E0 },
        { "photon_energy",                      photon_energy }
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