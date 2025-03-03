set(FOURIER_SOURCES
    ComplexFFT.cpp
    FFT.cpp
    FourierIntegral.cpp
)

add_library(Fourier STATIC ${FOURIER_SOURCES})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET_DEBUG_FLAGS(Fourier)
else()
    SET_COMPILER_FLAGS(Fourier)
endif()

target_include_directories(Fourier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(mrock REQUIRED)
find_package(FFTW3 REQUIRED)
find_library(FFTW3_LIB fftw3)

target_include_directories(Fourier PUBLIC ${mrock_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIRS})
target_link_libraries(Fourier PUBLIC ${mrock_LIBRARIES} ${FFTW3_LIB})