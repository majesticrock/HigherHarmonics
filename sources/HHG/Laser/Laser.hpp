#pragma once

#include "../GlobalDefinitions.hpp"
#include <cmath>
#include <array>
#include <concepts>

#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

namespace HHG::Laser {
    using Spline = boost::math::interpolators::cardinal_cubic_b_spline<h_float>;
    /**
    * We measure the time in units of the period of the electric field
    * and therefore the energy in units of hbar omega_L
    */
    struct Laser {
        const h_float momentum_amplitude{}; ///< e E_0 / (hbar omega_L)
        const h_float t_begin{}; ///< Only meaningful for pulsed lasers
        const h_float t_end{}; ///< Only meaningful for pulsed lasers
        const h_float photon_energy{}; ///< Not needed for the class itself, but saved for metadata purposes

        virtual ~Laser() = default;
        /**
         * @param photon_energy \f$ \hbar \omega_L \f$ in meV
         * @param E_0 peak electric field strength in MV / cm
         */
        Laser(h_float photon_energy, h_float E_0, h_float model_ratio);
        Laser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float t_begin, h_float t_end);
        Laser(h_float photon_energy, h_float E_0, h_float model_ratio, h_float t_begin, h_float t_end, bool _use_spline);
        
        virtual h_float envelope(h_float t) const = 0;
        inline h_float laser_function(h_float t) const {
            if (!use_spline)
                return momentum_amplitude * __laser_function__(t);
            if (t > t_end) return h_float{};
            return laser_spline(t);
        }
        inline h_float raw_laser_function(h_float t) const {
            if (!use_spline)
                return __laser_function__(t);
            if (t > t_end) return h_float{};
            return laser_spline(t) / momentum_amplitude;
        }

        std::array<h_float, 4> magnus_coefficients(h_float delta_t, h_float t_0) const;
  
    protected:
        Spline laser_spline;
        const bool use_spline{};

        virtual void compute_spline();
        inline h_float __laser_function__(h_float t) const {
            return envelope(t) * std::sin(t);
        }
    };
}