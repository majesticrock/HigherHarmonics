# Custom architecture target
set(TARGET_ARCH
    "native"
    CACHE STRING
    "Architecture passed to compiler as -march=<arch> (empty disables)"
)

# Joined compile options
add_library(hhg_options INTERFACE)
target_compile_features(hhg_options
    INTERFACE
        cxx_std_20
)

target_compile_options(hhg_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wextra>
)
if(TARGET_ARCH)
    target_compile_options(hhg_options INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-march=${TARGET_ARCH}>
    )
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hhg_options INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-ffast-math> 
    )
else()
    target_compile_definitions(hhg_options INTERFACE DEBUG)
    target_compile_options(hhg_options INTERFACE -g)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "NO_MPI")
    target_compile_definitions(hhg_options INTERFACE NO_MPI)
endif()

target_include_directories(hhg_options INTERFACE
    "$ENV{HOME}/usr/local/include"
)
