cmake_minimum_required(VERSION 3.30)

project(HigherHarmonics VERSION 2.1.0 LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/usr/local")

execute_process(
    COMMAND hostname
    OUTPUT_VARIABLE HOSTNAME
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(HOSTNAME STREQUAL "gw9.cluster.cl1")
# the new cluster needs help finding the correct version of boost
    set(Boost_DIR /usr/lib64/openmpi/lib/cmake/Boost-1.78.0)
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "NO_MPI")
    find_package(MPI REQUIRED)
endif()

find_package(OpenMP REQUIRED)
find_package(FFTW3 REQUIRED)
find_library(FFTW3_LIB fftw3)
find_package(mrock 
    REQUIRED 
    COMPONENTS 
        utility
)

# provided by the mrock library
mrock_generate_information_header(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/hhg/info.h"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUT_HEADER MROCK_GENERATED_INFO_HEADER
    OUT_INCLUDE_DIR MROCK_GENERATED_INCLUDE_DIR
)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/hhg_options.cmake)
# Sets include dir for the generated metadata header file
target_include_directories(hhg_options INTERFACE ${MROCK_GENERATED_INCLUDE_DIR})

add_executable(hhg 
    sources/hhg.cpp
)
target_sources(hhg PRIVATE
    sources/HHG/GeneralMagnus.cpp
    sources/HHG/TimeIntegrationConfig.cpp
    sources/HHG/GreensFunction.cpp
)

add_subdirectory(sources/HHG/Dispatch)
add_subdirectory(sources/HHG/Fourier)
add_subdirectory(sources/HHG/Systems)
add_subdirectory(sources/HHG/Laser)

if(NOT CMAKE_BUILD_TYPE STREQUAL "NO_MPI")
    target_link_libraries(hhg PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(hhg 
    PRIVATE 
        Dispatch 
        Fourier
        mrock::utility
        hhg_options
)
