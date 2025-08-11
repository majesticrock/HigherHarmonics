#pragma once
#include "../GlobalDefinitions.hpp"

#include <vector>
#include <utility>

namespace HHG {
    template<class T> using two_D_vector = std::vector<std::vector<T>>;

    struct OccupationContainer {
        typedef std::pair<h_float, h_float> occupation_t;

        two_D_vector<occupation_t> _data;
        const size_t N{};

        OccupationContainer(size_t _N)
            : _data(
                _N, std::vector<occupation_t>(
                    _N, occupation_t{0.0, 0.0}
                )
            ), N{_N}
        {}

        inline h_float& lower_band(size_t x, size_t z) {
            return _data[x][z].first;
        }
        inline h_float lower_band(size_t x, size_t z) const {
            return _data[x][z].first;
        }

        inline h_float& upper_band(size_t x, size_t z) {
            return _data[x][z].second;
        }
        inline h_float upper_band(size_t x, size_t z) const {
            return _data[x][z].second;
        }

        inline occupation_t& operator()(size_t x, size_t z) {
            return _data[x][z];
        }
        inline const occupation_t& operator()(size_t x, size_t z) const {
            return _data[x][z];
        }

        inline two_D_vector<h_float> entire_lower_band() const {
            two_D_vector<h_float> ret(N, std::vector<h_float>(N));
            for (size_t x = 0U; x < N; ++x) {
                for (size_t z = 0U; z < N; ++z) {
                    ret[x][z] = _data[x][z].first;
                }
            }
            return ret;
        }
        inline two_D_vector<h_float> entire_upper_band() const {
            two_D_vector<h_float> ret(N, std::vector<h_float>(N));
            for (size_t x = 0U; x < N; ++x) {
                for (size_t z = 0U; z < N; ++z) {
                    ret[x][z] = _data[x][z].second;
                }
            }
            return ret;
        }
    };
}