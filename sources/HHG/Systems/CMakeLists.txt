set(SYSTEMS_SOURCES
    DiracSystem.cpp
    DiracDetail/MagnusMatrix.cpp
    PiFlux.cpp
    Honeycomb.cpp
)

add_library(Systems STATIC ${SYSTEMS_SOURCES})

SET_COMPILER_FLAGS(Systems)

target_include_directories(Systems PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Systems PUBLIC OpenMP::OpenMP_CXX Laser)