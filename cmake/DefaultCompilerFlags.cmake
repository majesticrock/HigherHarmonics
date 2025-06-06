# DefaultCompilerFlags.cmake

function(SET_COMPILER_FLAGS TARGET)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
            message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} is not supported. GCC 11.0 or newer is required for C++20 support.")
        endif()

        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${TARGET} PRIVATE -Wall -Wno-sign-compare -fopenmp -march=native -O0 -g)# -fsanitize=undefined,address
            #target_link_options(${TARGET} PRIVATE -fsanitize=undefined,address)
        else()
            target_compile_options(${TARGET} PRIVATE -Wall -Wno-sign-compare -fopenmp -march=native -O3 -ffast-math)
            if(CMAKE_BUILD_TYPE STREQUAL "NDEBUG")
                target_compile_definitions(${TARGET} PRIVATE NDEBUG)
            endif()
            if(CMAKE_BUILD_TYPE STREQUAL "NO_MPI")
                target_compile_definitions(${TARGET} PRIVATE NO_MPI)
            endif()
        endif()
    else()
        message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}. Only GCC is supported.")
    endif()
endfunction()