# DefaultCompilerFlags.cmake

function(SET_COMPILER_FLAGS TARGET)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
            message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} is not supported. GCC 11.0 or newer is required for C++20 support.")
        endif()

        target_compile_options(${TARGET} PRIVATE -Wall -Wno-sign-compare -fopenmp -march=native -O3)
    else()
        message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}. Only GCC is supported.")
    endif()
endfunction()

function(SET_DEBUG_FLAGS TARGET)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
            message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} is not supported. GCC 11.0 or newer is required for C++20 support.")
        endif()

        target_compile_options(${TARGET} PRIVATE -Wall -Wno-sign-compare -fopenmp -march=native -O0 -g)
    else()
        message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}. Only GCC is supported.")
    endif()
endfunction()