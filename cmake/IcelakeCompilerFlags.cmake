# ClusterCompilerFlags.cmake

function(SET_COMPILER_FLAGS TARGET)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET} PRIVATE -Wall -Wno-sign-compare  -march=icelake-server -O3 -ffast-math)
        target_compile_definitions(${TARGET} PRIVATE NDEBUG MROCK_CL1)
    else()
        message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}. Only GCC|Clang is supported.")
    endif()
endfunction()