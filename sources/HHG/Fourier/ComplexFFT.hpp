#pragma once

#include "../GlobalDefinitions.hpp"
#include <fftw3.h>

namespace HHG::Fourier {
    struct ComplexFFT {
        fftw_plan plan;
        fftw_complex* in;
        // std::complex<double>* x can be cast to fftw_complex via reinterpret_cast<fftw_complex*>(x)
        fftw_complex* out;
        const int N;
        
        ComplexFFT() = delete;
        ComplexFFT(int _N);
        ~ComplexFFT(); ///< IMPORTANT! Frees allocated memory because C is difficult!
        
        void compute(const std::vector<h_complex>& input, std::vector<h_float>& real_output, std::vector<h_float>& imag_output);
        void compute(const std::vector<h_complex>& input, std::vector<h_complex>& output);
    };
}