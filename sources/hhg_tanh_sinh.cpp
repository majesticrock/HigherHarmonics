#include <iostream>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <chrono>
#include <numeric>

#include <nlohmann/json.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/InputFileReader.hpp>
#include <mrock/utility/better_to_string.hpp>
#include <mrock/utility/ElementwiseVector.hpp>

#include "HHG/DiracSystem.hpp"
#include "HHG/ContinuousLaser.hpp"
#include "HHG/FFT.hpp"
#include "HHG/WelchWindow.hpp"

using namespace HHG;

typedef boost::math::quadrature::tanh_sinh<HHG::h_float> integrator;

// k_z * kappa / |k| is found analytically (compare the HHG document)
inline HHG::h_float integration_weight(HHG::h_float k_z, HHG::h_float kappa) {
    return k_z * kappa / HHG::norm(k_z, kappa);
}

struct kappa_integrand {
    const h_float k_z{};
    const DiracSystem& system;
    const std::unique_ptr<Laser>& laser;
    const TimeIntegrationConfig& time_config;

    typedef mrock::utility::ElementwiseVector<std::vector<h_float>> result_type;

    kappa_integrand(h_float _k_z, const DiracSystem& _system, const std::unique_ptr<Laser>& _laser, const TimeIntegrationConfig& _time_config)
        : k_z(_k_z), system(_system), laser(_laser), time_config(_time_config) {}

    result_type operator()(h_float kappa) const {
        result_type rhos_buffer(time_config.n_measurements, h_float{}, result_type::allocator_type(), mrock::utility::L2SquaredNorm());
        
        system.time_evolution_sigma(rhos_buffer.elements, laser.get(), k_z, kappa, time_config);
        rhos_buffer *= integration_weight(k_z, kappa);
        return rhos_buffer;
    }
};

struct k_z_integrand {
    const DiracSystem& system;
    const std::unique_ptr<Laser>& laser;
    const TimeIntegrationConfig& time_config;

    using result_type = kappa_integrand::result_type;
    
    k_z_integrand(const DiracSystem& _system, const std::unique_ptr<Laser>& _laser, const TimeIntegrationConfig& _time_config)
        : system(_system), laser(_laser), time_config(_time_config) {}

    result_type operator()(h_float k_z) const {
        integrator kappa_integrator;
        kappa_integrand m_kappa_integrand(k_z, system, laser, time_config);
        const auto up = system.kappa_integration_upper_limit(k_z);
        if (is_zero(up) ) return result_type(time_config.n_measurements, h_float{});

        double termination = 4000 * std::sqrt(std::numeric_limits<double>::epsilon());
        double error;
        double L1;
        size_t levels;
        auto Q = kappa_integrator.integrate(m_kappa_integrand, h_float{0}, system.kappa_integration_upper_limit(k_z), 
                    termination, &error, &L1, &levels);
        double condition_number = L1/abs(Q);
        std::cout << "Inner condition_number " << condition_number << std::endl;
        return Q;
    }
};

#pragma omp declare reduction(vec_plus : std::vector<HHG::h_float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<HHG::h_float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

int main(int argc, char** argv) {
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

    high_resolution_clock::time_point begin = high_resolution_clock::now();
    std::cout << "Computing the k integrals..." << std::endl;

    integrator z_integrator;
    k_z_integrand m_k_z_integrand(system, laser, time_config);
    auto current_density_time = z_integrator.integrate(m_k_z_integrand, -system.convert_to_z_integration(1), system.convert_to_z_integration(1));

    high_resolution_clock::time_point end = high_resolution_clock::now();
	std::cout << "Runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    begin = high_resolution_clock::now();
    std::cout << "Computing the FFT..." << std::endl;

    FFT fft(N);

    WelchWindow window(N);
    for (int i = 0; i < N; ++i) {
        current_density_time[i] *= window[i];
    }
    std::vector<h_float> current_density_frequency_real(N);
    std::vector<h_float> current_density_frequency_imag(N);
    fft.compute(current_density_time.elements, current_density_frequency_real, current_density_frequency_imag);

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
        { "t_begin",                            time_config.t_begin },
        { "t_end",                              time_config.t_end },
        { "n_laser_cycles",                     n_laser_cylces },
        { "n_measurements",                     time_config.n_measurements },
        { "current_density_time",               current_density_time.elements },
        { "current_density_frequency_real",     current_density_frequency_real },
        { "current_density_frequency_imag",     current_density_frequency_imag },
        { "T",                                  temperature },
        { "E_F",                                E_F },
        { "v_F",                                v_F },
        { "band_width",                         band_width },
        { "field_amplitude",                    E0 },
        { "photon_energy",                      photon_energy },
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