#include "FFT.hpp"
#include <cassert>
#include <mrock/utility/ComplexNumberIterators.hpp>

namespace HHG::Fourier {
    FFT::FFT(int _N)
        : N{_N}
    {
        in = (h_float*) fftw_malloc(sizeof(h_float) * N);
        out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
        // FFTW_MEASURE is good if many FFTs of data sets of the same size are needed.
        // Otherwise we should use FFTW_ESTIMATE.
        // I want to test this, but I assume that FFTW_MESAURE will be superior in our case.
        // The plan must be created before we set the input and output.
        plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_MEASURE);
    }

    FFT::~FFT()
    {
        // IMPORTANT!
        fftw_destroy_plan(plan);
        fftw_free(in); 
        fftw_free(out);
    }

    void FFT::compute(const std::vector<h_float>& input, std::vector<h_float>& real_output, std::vector<h_float>& imag_output)
    {
        assert(input.size() >= N);
        std::copy(input.begin(), input.end(), in);
        fftw_execute(plan);

        real_output.resize(N / 2 + 1);
        imag_output.resize(N / 2 + 1);

        auto real_begin = mrock::utility::make_real_part_iterator(reinterpret_cast<h_complex*>(out));
        auto imag_begin = mrock::utility::make_imag_part_iterator(reinterpret_cast<h_complex*>(out));
        auto real_end = mrock::utility::make_real_part_iterator_end(reinterpret_cast<h_complex*>(out), N / 2 + 1);
        auto imag_end = mrock::utility::make_imag_part_iterator_end(reinterpret_cast<h_complex*>(out), N / 2 + 1);
        
        std::copy(real_begin, real_end, real_output.begin());
        std::copy(imag_begin, imag_end, imag_output.begin());
    }
}