set(LASER_SOURCES
    ContinuousLaser.cpp
    CosineLaser.cpp
    Laser.cpp
)

add_library(Laser STATIC ${LASER_SOURCES})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET_DEBUG_FLAGS(Laser)
else()
    SET_COMPILER_FLAGS(Laser)
endif()

target_include_directories(Laser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})