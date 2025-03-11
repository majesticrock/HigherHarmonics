cmake_minimum_required(VERSION 3.16)

project(HigherHarmonics)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH "~/usr/local")
include_directories(
    $ENV{HOME}/usr/local/include
)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(mrock REQUIRED)

if(CLUSTER_BUILD)
    include(${CMAKE_SOURCE_DIR}/cmake/ClusterCompilerFlags.cmake)
else()
    include(${CMAKE_SOURCE_DIR}/cmake/DefaultCompilerFlags.cmake)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/../../PhdUtility/cmake/mrock-information.cmake)

add_subdirectory(sources/HHG/Laser)
add_subdirectory(sources/HHG/Fourier)

add_executable(hhg 
    sources/hhg.cpp
)
target_sources(hhg PRIVATE
    sources/HHG/DiracSystem.cpp
    sources/HHG/TimeIntegrationConfig.cpp
    sources/HHG/GreensFunction.cpp
)

SET_COMPILER_FLAGS(hhg)

if(NOT CMAKE_BUILD_TYPE STREQUAL "NO_MPI")
    target_link_libraries(hhg PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(hhg PRIVATE Laser Fourier OpenMP::OpenMP_CXX ${mrock_LIBRARIES})
target_include_directories(hhg PRIVATE ${mrock_INCLUDE_DIRS})