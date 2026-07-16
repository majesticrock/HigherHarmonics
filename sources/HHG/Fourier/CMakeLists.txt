set(FOURIER_SOURCES
    ComplexFFT.cpp
    FFT.cpp
    FourierIntegral.cpp
    TrapezoidalFFT.cpp
)

add_library(Fourier STATIC ${FOURIER_SOURCES})

target_include_directories(Fourier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Fourier PUBLIC mrock::utility hhg_options ${FFTW3_LIB})