if(NOT DEFINED CLUSTER_BUILD)
    set(CLUSTER_BUILD "default" CACHE STRING "Choose CPU target: default, cascadelake, icelake")
else()
    set(valid_cluster_builds "default" "cascadelake" "icelake")
    list(FIND valid_cluster_builds "${CLUSTER_BUILD}" cluster_build_index)
    if(cluster_build_index EQUAL -1)
        message(FATAL_ERROR "Invalid value for CLUSTER_BUILD: ${CLUSTER_BUILD}. Valid options are: ${valid_cluster_builds}")
    endif()
    set(CLUSTER_BUILD "${CLUSTER_BUILD}" CACHE STRING "Choose CPU target: default, cascadelake, icelake" FORCE)
endif()

set_property(CACHE CLUSTER_BUILD PROPERTY STRINGS "default" "cascadelake" "icelake")

if(CLUSTER_BUILD STREQUAL "default")
    include(${CMAKE_SOURCE_DIR}/cmake/DefaultCompilerFlags.cmake)
elseif(CLUSTER_BUILD STREQUAL "cascadelake")
    #set(Boost_DIR /usr/lib64/openmpi/lib/cmake/Boost-1.78.0)# the new cluster needs help finding the correct version of boost
    include(${CMAKE_SOURCE_DIR}/cmake/CascadelakeCompilerFlags.cmake)
elseif(CLUSTER_BUILD STREQUAL "icelake")
    set(Boost_DIR /usr/lib64/openmpi/lib/cmake/Boost-1.78.0)# the new cluster needs help finding the correct version of boost
    include(${CMAKE_SOURCE_DIR}/cmake/IcelakeCompilerFlags.cmake)
endif()