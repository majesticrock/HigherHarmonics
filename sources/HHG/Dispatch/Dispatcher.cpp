#include "Dispatcher.hpp"

namespace HHG::Dispatch {
    nlohmann::json Dispatcher::special_information() const
    {
        return nlohmann::json{};
    }
}