#include "ExperimentalLaser.hpp"
#include <iostream>

namespace HHG::Laser {
    ExperimentalLaser::ExperimentalLaser(h_float _photon_energy, h_float _E_0, h_float model_ratio, h_float _second_laser_shift, Active _active_laser/* = Active::Both */)
        : Laser(_photon_energy * exp_photon_energy, 
            _E_0, 
            model_ratio, 
            0, 
            unified_t_max * (_photon_energy * exp_photon_energy / (1e12 * hbar)), //(laser_end + std::abs(_second_laser_shift)) * (_photon_energy * exp_photon_energy / (1e12 * hbar)), 
            true), 
        second_laser_shift{_second_laser_shift * (_photon_energy * exp_photon_energy / (1e12 * hbar))},
        active_laser{_active_laser}
    {
        this->compute_spline();
    }

    std::array<h_float, ExperimentalLaser::N_experiment + 2 * ExperimentalLaser::N_extra> vector_potential(const std::array<h_float, ExperimentalLaser::N_experiment>& electric_field, h_float dt) 
    {
        std::array<h_float, ExperimentalLaser::N_experiment + 2 * ExperimentalLaser::N_extra> ret;
        ret[ExperimentalLaser::N_extra] = -electric_field[0] * dt;
        for (size_t i = 1U; i < ExperimentalLaser::N_experiment; ++i) {
            ret[ExperimentalLaser::N_extra + i] = ret[ExperimentalLaser::N_extra + i - 1] - electric_field[i] * dt; // A = - 1/c int_0^t E(t') dt'. The factor 1/c cancels in the Peierls substitution   
        }

        {
            constexpr size_t avg_length = 32U;
            h_float end_avg{};
            for (size_t i = ExperimentalLaser::N_experiment - avg_length; i < ExperimentalLaser::N_experiment; ++i) {
                end_avg += ret[ExperimentalLaser::N_extra + i];
            }
            end_avg /= avg_length;
            for (size_t i = 1U; i < ExperimentalLaser::N_experiment; ++i) {
                ret[ExperimentalLaser::N_extra + i] -= 0.98 * end_avg;
            }
        }

        {
            constexpr size_t true_begin = 36U;
            constexpr h_float b = 3e-3;
            for (size_t i = 0U; i < true_begin; ++i) {
                const h_float x = static_cast<h_float>(i) - static_cast<h_float>(true_begin);
                ret[ExperimentalLaser::N_extra + i] *= std::exp(-b*x*x);
            }
        }

        {
            const h_float prime = -electric_field.front();
            const h_float primeprime = -0.5 * (electric_field[1] - electric_field.front()) / dt;

            const h_float __end = -(ExperimentalLaser::N_extra - 1) * dt;
            const h_float third = -4 * (primeprime / (2 * __end) + (3 * prime) / (4 * __end * __end) + ret[ExperimentalLaser::N_extra] / (__end * __end * __end));
            const h_float fourth = -0.25 * (3 * third / __end + 2 * primeprime / (__end * __end) + prime / (__end * __end * __end));

            auto pol = [&](h_float t) {
                return ret[ExperimentalLaser::N_extra] + prime * t + primeprime * t*t + third * t*t*t + fourth * t*t*t*t;
            };

            for (int i = 1; i <= ExperimentalLaser::N_extra; ++i) {
                ret[ExperimentalLaser::N_extra - i] = pol(-i*dt);
            }
            ret.front() = h_float{};
        }

        {
            const h_float prime = -electric_field.back();
            const h_float primeprime = 0.5 * (electric_field[ExperimentalLaser::N_experiment - 2] - electric_field.back()) / dt;

            const h_float __end = (ExperimentalLaser::N_extra - 1) * dt;
            const h_float third = -4 * (primeprime / (2 * __end) + (3 * prime) / (4 * __end * __end) + ret[ExperimentalLaser::N_extra + ExperimentalLaser::N_experiment - 1] / (__end * __end * __end));
            const h_float fourth = - 0.25 * (3 * third / __end + 2 * primeprime / (__end * __end) + prime / (__end * __end * __end));

            auto pol = [&](h_float t) {
                return ret[ExperimentalLaser::N_extra + ExperimentalLaser::N_experiment - 1] + prime * t + primeprime * t*t + third * t*t*t + fourth * t*t*t*t;
            };

            for (int i = 0; i < (ExperimentalLaser::N_extra - 1); ++i) {
                ret[ExperimentalLaser::N_extra + ExperimentalLaser::N_experiment + i] = pol((i+1)*dt);
            }
            ret.back() = h_float{};
        }
        return ret;
    }

    void ExperimentalLaser::compute_spline()
    {
        constexpr int N = N_experiment + 2 * N_extra;
        const h_float dt = this->photon_energy * (exp_dt / (1e12 * hbar)); // unitless
        const h_float unitless_laser_end = this->photon_energy * (laser_end / (1e12 * hbar));

        // Experimental data in MV/cm [measured]
        constexpr std::array<h_float, N_experiment> E_A = { -2.55696e-3, 2.10912e-3, 6.31776e-3, 9.55008e-3, 12.92784e-3, 14.19168e-3, 14.73648e-3, 12.94512e-3, 10.848e-3, 7.81296e-3, 5.56224e-3, 2.75136e-3, 0.94992e-3, 0.90144e-3, -0.00576e-3, -0.47712e-3, 1.47744e-3, 3.53376e-3, 6.2232e-3, 9.19584e-3, 10.59264e-3, 11.9496e-3, 13.90368e-3, 14.79456e-3, 14.42928e-3, 14.69664e-3, 12.96336e-3, 12.09024e-3, 10.44288e-3, 9.5544e-3, 9.50688e-3, 9.324e-3, 9.44592e-3, 9.80448e-3, 10.42848e-3, 10.14624e-3, 10.71744e-3, 10.74144e-3, 11.29392e-3, 12.27216e-3, 14.484e-3, 15.33264e-3, 16.37376e-3, 17.77248e-3, 17.4864e-3, 17.02848e-3, 14.20224e-3, 10.47888e-3, 4.25232e-3, -2.6328e-3, -10.33536e-3, -18.71472e-3, -28.8456e-3, -39.21744e-3, -49.02864e-3, -63.51024e-3, -78.50592e-3, -95.61984e-3, -114.37488e-3, -137.49504e-3, -159.21216e-3, -179.00112e-3, -198.012e-3, -210.59136e-3, -220.03488e-3, -217.95456e-3, -207.36384e-3, -185.70384e-3, -150.52896e-3, -95.31168e-3, -25.4928e-3, 65.42256e-3, 174.52992e-3, 301.66368e-3, 452.8152e-3, 620.43168e-3, 764.65248e-3, 902.82144e-3, 986.56944e-3, 1036.5168e-3, 1023.91392e-3, 931.74576e-3, 728.73312e-3, 353.4624e-3, -51.29184e-3, -497.22672e-3, -1008.81456e-3, -1390.3368e-3, -1563.96912e-3, -1578.59424e-3, -1425.19296e-3, -1124.78544e-3, -653.9136e-3, -137.19216e-3, 241.85616e-3, 541.9464e-3, 690.40512e-3, 681.43872e-3, 542.65248e-3, 317.91408e-3, 54.77472e-3, -186.63648e-3, -336.83328e-3, -388.92336e-3, -341.38896e-3, -243.9744e-3, -100.69536e-3, 20.13264e-3, 105.09984e-3, 134.44896e-3, 106.96272e-3, 60.70128e-3, 6.39168e-3, -23.56848e-3, -18.42432e-3, 11.94816e-3, 66.47472e-3, 130.02624e-3, 183.82416e-3, 214.10976e-3, 217.644e-3, 190.53696e-3, 146.1816e-3, 93.20928e-3, 41.51232e-3, 7.55904e-3, -7.24464e-3, -10.84992e-3, -4.59792e-3, -3.02448e-3, -3.48192e-3, -14.8752e-3, -32.49744e-3, -53.08608e-3, -67.54848e-3, -68.22432e-3, -54.75168e-3, -28.3728e-3, 3.53904e-3, 38.86608e-3, 67.60512e-3, 84.57072e-3, 87.24864e-3, 72.96432e-3, 46.76688e-3, 12.67872e-3, -22.67856e-3, -54.4056e-3, -75.13104e-3, -84.07104e-3, -82.01712e-3, -68.90736e-3, -45.13728e-3, -15.16704e-3, 15.20496e-3, 41.75472e-3, 63.07248e-3, 73.20096e-3, 75.97104e-3, 68.9568e-3, 56.25696e-3, 39.71136e-3, 22.404e-3, 9.47136e-3, 1.56864e-3, -0.20832e-3, 1.5264e-3, 3.66528e-3, 4.70736e-3, 2.73552e-3, -4.04112e-3, -13.6104e-3, -22.5816e-3, -29.51664e-3, -30.87696e-3, -25.97952e-3, -15.42384e-3, -0.50544e-3, 11.41056e-3, 19.54464e-3, 22.33536e-3, 16.58208e-3, 4.84992e-3, -12.80976e-3, -30.85488e-3, -43.15872e-3, -50.3448e-3, -47.31744e-3, -37.52448e-3, -23.10048e-3, -7.1544e-3, 10.32384e-3, 21.94032e-3, 29.208e-3, 32.27232e-3, 33.51888e-3, 31.52064e-3, 29.2632e-3, 26.72112e-3, 23.49840e-3, 17.39232e-3 };
        // Experimental data in MV/cm [measured]
        constexpr std::array<h_float, N_experiment> E_B = { -2.89152e-3, -2.0304e-3, 0.06384e-3, 2.33424e-3, 4.02384e-3, 4.99344e-3, 4.95648e-3, 4.07136e-3, 2.6232e-3, 0.18384e-3, -1.07424e-3, -1.30944e-3, -3.4824e-3, -2.44992e-3, -3.16272e-3, -4.13904e-3, -3.62064e-3, -3.64944e-3, -2.40672e-3, -2.2488e-3, -0.94848e-3, -0.4896e-3, 1.61856e-3, 2.02896e-3, 3.22416e-3, 2.37504e-3, 2.2656e-3, 2.60688e-3, 1.66128e-3, 1.31856e-3, 1.13136e-3, 2.076e-3, 3.10464e-3, 3.78624e-3, 3.88464e-3, 4.29696e-3, 2.03184e-3, 1.17504e-3, -1.39296e-3, -2.99136e-3, -2.91072e-3, -2.77056e-3, -2.19456e-3, -0.33168e-3, 1.71264e-3, 2.93904e-3, 3.15072e-3, 2.71968e-3, 0.82608e-3, -1.53504e-3, -3.80208e-3, -7.18128e-3, -10.90176e-3, -14.62752e-3, -16.93536e-3, -20.98656e-3, -25.18032e-3, -29.35344e-3, -33.63696e-3, -40.9176e-3, -46.9464e-3, -52.05312e-3, -57.47424e-3, -61.1784e-3, -62.55936e-3, -61.89072e-3, -58.7088e-3, -49.87632e-3, -39.12816e-3, -21.95664e-3, 0.3288e-3, 32.1336e-3, 65.65056e-3, 107.80848e-3, 155.69184e-3, 206.0256e-3, 250.992e-3, 299.1e-3, 329.27952e-3, 344.52528e-3, 334.83072e-3, 292.75824e-3, 216.2976e-3, 87.13968e-3, -60.76896e-3, -213.16272e-3, -381.44928e-3, -550.64592e-3, -668.63328e-3, -697.77456e-3, -614.68416e-3, -474.13776e-3, -295.02624e-3, -101.68176e-3, 77.6952e-3, 229.9536e-3, 325.01232e-3, 366.3264e-3, 347.13792e-3, 286.46496e-3, 190.91664e-3, 87.9312e-3, 3.15264e-3, -52.75632e-3, -73.94208e-3, -69.7128e-3, -44.42496e-3, -15.6192e-3, 9.04512e-3, 18.3912e-3, 11.7744e-3, -3.59952e-3, -24.15888e-3, -37.5696e-3, -40.3272e-3, -32.69184e-3, -14.71968e-3, 10.49088e-3, 31.34592e-3, 44.9976e-3, 49.40496e-3, 41.9376e-3, 26.91168e-3, 6.828e-3, -12.80208e-3, -27.65088e-3, -33.31008e-3, -33.5112e-3, -27.49104e-3, -20.3928e-3, -12.38784e-3, -7.29312e-3, -4.30176e-3, -4.9032e-3, -6.6192e-3, -5.38032e-3, -2.08416e-3, 4.74816e-3, 13.29168e-3, 24.12864e-3, 33.80832e-3, 39.65472e-3, 41.70384e-3, 37.39728e-3, 28.0104e-3, 14.1456e-3, -0.42528e-3, -14.8632e-3, -25.93872e-3, -32.73024e-3, -35.59632e-3, -33.16224e-3, -26.81424e-3, -15.35616e-3, -3.79344e-3, 7.71312e-3, 17.02512e-3, 20.75664e-3, 21.26784e-3, 18.18288e-3, 10.12464e-3, 1.44624e-3, -6.43872e-3, -11.79696e-3, -15.60672e-3, -15.07008e-3, -11.30304e-3, -6.09888e-3, -0.54e-3, 3.9984e-3, 5.51328e-3, 7.15632e-3, 5.95824e-3, 3.05376e-3, 0.89424e-3, -0.03504e-3, 1.70256e-3, 4.47936e-3, 3.20112e-3, 4.31088e-3, 3.30192e-3, -1.64208e-3, -6.57648e-3, -14.65632e-3, -21.73104e-3, -26.52048e-3, -28.66992e-3, -25.71216e-3, -19.25712e-3, -10.89264e-3, -2.78688e-3, 8.28048e-3, 14.63712e-3, 19.45968e-3, 20.5056e-3, 19.42896e-3, 17.1672e-3, 13.2096e-3, 9.61536e-3, 6.31152e-3, 2.26608e-3 };

        Spline __spline_A(vector_potential(E_A, dt).data(), N, t_begin, dt, h_float{}, h_float{});
        Spline __spline_B(vector_potential(E_B, dt).data(), N, t_begin, dt, h_float{}, h_float{});

        auto add_laser = [this](const h_float A_A, const h_float A_B) -> h_float {
            if (this->active_laser == Active::Both)
                return A_A + A_B;
            if (this->active_laser == Active::A)
                return A_A;
            else
                return A_B;
        };

        const int N_temp = t_end / dt + 1;//N + int(second_laser_shift / dt) + 1;
        std::vector<h_float> __temp(N_temp + 1);
        for(int i = 0; i <= N_temp; ++i) {
            const h_float t = t_begin + (dt * i);
            //__temp[i] = __spline_A(t) + __spline_B(t);
            if (second_laser_shift >= 0) {
                const h_float __A = t <= unitless_laser_end ? __spline_A(t) : h_float{};
                const h_float __B = (t - second_laser_shift >= 0 && t - second_laser_shift <= unitless_laser_end) ? __spline_B(t - second_laser_shift) : h_float{};
                __temp[i] = momentum_amplitude * add_laser(__A, __B);
            }
            else {
                const h_float __A = (t + second_laser_shift >= 0 && t + second_laser_shift <= unitless_laser_end) ? __spline_A(t + second_laser_shift) : h_float{};
                const h_float __B = t <= unitless_laser_end ? __spline_B(t) : h_float{};
                __temp[i] = momentum_amplitude * add_laser(__A, __B);
            }
        }

        this->laser_spline = Spline(__temp.data(), N_temp + 1, t_begin, dt, h_float{}, h_float{});
    }

    h_float ExperimentalLaser::envelope(h_float t) const {
        throw std::runtime_error("Enevelope of the experimental pulse should never be called!");
    }
}