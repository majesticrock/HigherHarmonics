#include "ComplexFFT.hpp"
#include <cassert>
#include <mrock/utility/ComplexNumberIterators.hpp>

namespace HHG::Fourier {
    ComplexFFT::ComplexFFT(int _N)
        : N{_N}
    {
        in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        // FFTW_MEASURE is good if many FFTs of data sets of the same size are needed.
        // Otherwise we should use FFTW_ESTIMATE.
        // I want to test this, but I assume that FFTW_MESAURE will be superior in our case.
        // The plan must be created before we set the input and output.
        plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
    }

    ComplexFFT::~ComplexFFT()
    {
        // IMPORTANT!
        fftw_destroy_plan(plan);
        fftw_free(in); 
        fftw_free(out);
    }

    void ComplexFFT::compute(const std::vector<h_complex>& input, std::vector<h_float>& real_output, std::vector<h_float>& imag_output)
    {
        assert(input.size() >= N);
        std::copy(input.begin(), input.end(), reinterpret_cast<h_complex*>(in));
        fftw_execute(plan);

        real_output.resize(N);
        imag_output.resize(N);

        auto real_begin = mrock::utility::make_real_part_iterator(reinterpret_cast<h_complex*>(out));
        auto imag_begin = mrock::utility::make_imag_part_iterator(reinterpret_cast<h_complex*>(out));
        auto real_end = mrock::utility::make_real_part_iterator_end(reinterpret_cast<h_complex*>(out), N);
        auto imag_end = mrock::utility::make_imag_part_iterator_end(reinterpret_cast<h_complex*>(out), N);
        
        std::copy(real_begin, real_end, real_output.begin());
        std::copy(imag_begin, imag_end, imag_output.begin());
    }

    void ComplexFFT::compute(const std::vector<h_complex> &input, std::vector<h_complex> &output)
    {
        assert(input.size() == N);
        std::copy(input.begin(), input.end(), reinterpret_cast<h_complex*>(in));
        fftw_execute(plan);

        output.resize(N);
        std::copy_n(reinterpret_cast<h_complex*>(out), N, output.begin());
    }
}