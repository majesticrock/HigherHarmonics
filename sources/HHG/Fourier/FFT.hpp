#pragma once

#include "../GlobalDefinitions.hpp"
#include <fftw3.h>

namespace HHG::Fourier {
    struct FFT {
        fftw_plan plan;
        h_float* in;
        // std::complex<double>* x can be cast to fftw_complex via reinterpret_cast<fftw_complex*>(x)
        fftw_complex* out;
        const int N;
        
        FFT() = delete;
        FFT(int _N);
        ~FFT(); ///< IMPORTANT! Frees allocated memory because C is difficult!
        
        void compute(const std::vector<h_float>& input, std::vector<h_float>& real_output, std::vector<h_float>& imag_output);
    };
}