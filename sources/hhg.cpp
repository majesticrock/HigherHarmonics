#include <iostream>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <mrock/utility/OutputConvenience.hpp>
#include <mrock/utility/InputFileReader.hpp>

#include "HHG/DiracSystem.hpp"
#include "HHG/ContinuousLaser.hpp"
#include "HHG/FFT.hpp"

int main(int argc, char** argv) {
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
}