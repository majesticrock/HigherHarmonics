cmake_minimum_required(VERSION 3.16)

project(HigherHarmonics VERSION 2.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH "~/usr/local")
include_directories(
    $ENV{HOME}/usr/local/include
)

execute_process(
    COMMAND hostname
    OUTPUT_VARIABLE HOSTNAME
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(HOSTNAME STREQUAL "gw9.cluster.cl1")
    set(Boost_DIR /usr/lib64/openmpi/lib/cmake/Boost-1.78.0)# the new cluster needs help finding the correct version of boost
endif()

include(${CMAKE_SOURCE_DIR}/cmake/SelectClusterBuildOptions.cmake)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(FFTW3 REQUIRED)
find_library(FFTW3_LIB fftw3)
find_package(mrock REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../PhdUtility/cmake/mrock-information.cmake)

add_executable(hhg 
    sources/hhg.cpp
)
target_sources(hhg PRIVATE
    sources/HHG/GeneralMagnus.cpp
    sources/HHG/TimeIntegrationConfig.cpp
    sources/HHG/GreensFunction.cpp
)
SET_COMPILER_FLAGS(hhg)

add_subdirectory(sources/HHG/Dispatch)
add_subdirectory(sources/HHG/Fourier)
add_subdirectory(sources/HHG/Systems)
add_subdirectory(sources/HHG/Laser)

if(NOT CMAKE_BUILD_TYPE STREQUAL "NO_MPI")
    target_link_libraries(hhg PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(hhg PRIVATE Dispatch Fourier ${mrock_LIBRARIES})
target_include_directories(hhg PRIVATE ${mrock_INCLUDE_DIRS})