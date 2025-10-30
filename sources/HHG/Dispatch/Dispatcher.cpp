#include "Dispatcher.hpp"
#include "../Laser/ExperimentParams.hpp"

namespace HHG::Dispatch {
    nlohmann::json Dispatcher::special_information() const
    {
        return nlohmann::json{{"special_information", "empty"}};
    }

    h_float Dispatcher::get_photon_energy(mrock::utility::InputFileReader &input)
    {
        const auto laser_type = input.getString("laser_type");
        if (laser_type == exp || laser_type == expA || laser_type == expB)
            return input.getDouble("photon_energy") * Laser::exp_photon_energy;
        if (laser_type == dcos || laser_type == dcosA || laser_type == dcosB)
            return input.getDouble("photon_energy") * Laser::exp_photon_energy;
        if (laser_type == dgauss || laser_type == dgaussA || laser_type == dgaussB)
            return input.getDouble("photon_energy") * Laser::exp_photon_energy;
        return input.getDouble("photon_energy");
    }
}