set(DISPATCH_SOURCES
    DiracDispatcher.cpp
    PiFluxDispatcher.cpp
)

add_library(Dispatch STATIC ${DISPATCH_SOURCES})

SET_COMPILER_FLAGS(Dispatch)

target_include_directories(Dispatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE STREQUAL "NO_MPI")
    target_link_libraries(Dispatch PUBLIC MPI::MPI_CXX)
endif()

target_link_libraries(Dispatch PUBLIC Laser Systems)