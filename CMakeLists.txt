cmake_minimum_required(VERSION 3.16)

project(HigherHarmonics)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH "~/usr/local")
include_directories(
    $ENV{HOME}/usr/local/include
)

add_executable(hhg 
    sources/hhg.cpp
)

if(CLUSTER_BUILD)
    include(${CMAKE_SOURCE_DIR}/cmake/ClusterCompilerFlags.cmake)
else()
    include(${CMAKE_SOURCE_DIR}/cmake/DefaultCompilerFlags.cmake)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/../../PhdUtility/cmake/mrock-information.cmake)

SET_COMPILER_FLAGS(hhg)

find_package(OpenMP REQUIRED)
find_package(mrock REQUIRED)

target_link_libraries(hhg PRIVATE OpenMP::OpenMP_CXX ${mrock_LIBRARIES})
target_include_directories(hhg PRIVATE ${mrock_INCLUDE_DIRS})