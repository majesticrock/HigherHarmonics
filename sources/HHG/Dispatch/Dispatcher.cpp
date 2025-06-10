#include "Dispatcher.hpp"
#include "../Laser/ExperimentalLaser.hpp"

namespace HHG::Dispatch {
    nlohmann::json Dispatcher::special_information() const
    {
        return nlohmann::json{};
    }

    h_float Dispatcher::get_photon_energy(mrock::utility::InputFileReader &input)
    {
        const auto laser_type = input.getString("laser_type");
        if (laser_type == "exp" || laser_type == "expA" || laser_type == "expB")
            return input.getDouble("photon_energy") * Laser::ExperimentalLaser::exp_photon_energy;
        return input.getDouble("photon_energy");
    }
}