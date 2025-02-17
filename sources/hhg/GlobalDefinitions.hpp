#pragma once
#define _USE_MATH_DEFINES

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <stdint.h>
#include <limits>

namespace HHG {
    using h_float = double;
    using h_complex = std::complex<h_float>;

    template<Eigen::Index nrows, Eigen::Index ncols> using real_matrix = Eigen::Matrix<h_float, nrows, ncols>;
    template<Eigen::Index nrows> using real_vector = Eigen::Vector<h_float, nrows>;
    
    template<Eigen::Index nrows, Eigen::Index ncols> using complex_matrix = Eigen::Matrix<h_complex, nrows, ncols>;
    template<Eigen::Index nrows> using complex_vector = Eigen::Vector<h_complex, nrows>;

    constexpr h_complex imaginary_unit{0., 1.};
    constexpr h_float hbar = h_float(6.582119569509065698e-13); // meV s
    constexpr h_float pi = h_float(M_PI);

    template<class... Args>
    h_float norm(Args... args) {
        return std::sqrt((... + (args * args)));
    }

    /* This function abuses the structure of our desired precision:
	*  The mantissa is empty, i.e., we can solely rely on the exponent.
	*  If the exponent of <number> is >= 0b01111010011, |number| >= precision, i.e., not 0
	*/
	inline bool is_zero(double number) {
		static_assert(std::numeric_limits<double>::is_iec559, "IEEE 754 floating point not verified!");
		// 5.684341886080802e-14   <->   0 | 01111010011 | 0000000000000000000000000000000000000000000000000000
		uint64_t tmp; // silence compiler warnings (at 0 overhead)
		std::memcpy(&tmp, &number, sizeof(tmp));
		return static_cast<uint16_t>((tmp >> 52) & 0x7FF) < 0b01111010011;
	}

	inline bool is_zero(std::complex<double> number) {
		return is_zero(std::abs(number));
	}

}