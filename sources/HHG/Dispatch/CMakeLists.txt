set(DISPATCH_SOURCES
    DiracDispatcher.cpp
    PiFluxDispatcher.cpp
    HoneycombDispatcher.cpp
    Dispatcher.cpp
)

add_library(Dispatch STATIC ${DISPATCH_SOURCES})

target_include_directories(Dispatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(USE_MPI)
    target_link_libraries(Dispatch PUBLIC MPI::MPI_CXX)
endif()

target_link_libraries(Dispatch PUBLIC Laser Systems hhg_options)