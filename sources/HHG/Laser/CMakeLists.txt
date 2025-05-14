set(LASER_SOURCES
    ContinuousLaser.cpp
    CosineLaser.cpp
    Laser.cpp
)

add_library(Laser STATIC ${LASER_SOURCES})

SET_COMPILER_FLAGS(Laser)

target_include_directories(Laser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})