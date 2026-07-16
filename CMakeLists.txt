cmake_minimum_required(VERSION 3.30)

project(HigherHarmonics VERSION 2.1.0 LANGUAGES CXX)

#
# Set local settings and set the default build type as Release
#

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/usr/local")
set(Boost_DIR /usr/lib64/openmpi/lib/cmake/Boost-1.78.0)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type is '${CMAKE_BUILD_TYPE}'.")

#
# Library specifics
#

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(USE_MPI OFF)
else()
    option(USE_MPI
        "Use OpenMPI for parallelization"
        ON
    )
endif()

if(USE_MPI)
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

if(USE_MPI)
    target_link_libraries(hhg PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(hhg 
    PRIVATE 
        Dispatch 
        Fourier
        mrock::utility
        hhg_options
)
