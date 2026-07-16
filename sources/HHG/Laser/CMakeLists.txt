set(LASER_SOURCES
    ContinuousLaser.cpp
    CosineLaser.cpp
    Laser.cpp
    ExperimentalLaser.cpp
    QuenchedField.cpp
    PowerLawField.cpp
    GaussLaser.cpp
    #DoubleCosine.cpp
)

add_library(Laser STATIC ${LASER_SOURCES})

target_include_directories(Laser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Laser PRIVATE hhg_options)