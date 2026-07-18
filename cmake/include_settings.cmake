# cmake/user_settings.cmake
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/../../.mrock")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/usr/local")

set(EXTRA_INCLUDE_DIRS "$ENV{HOME}/usr/local/include" "${PROJECT_SOURCE_DIR}/../../.mrock/include")

set(USER_CONFIG_FILE "${CMAKE_CURRENT_LIST_DIR}/user_settings.cmake")
if(EXISTS "USER_CONFIG_FILE")
    mrock_message("Loading user configuration: ${USER_CONFIG_FILE}")
    include("${USER_CONFIG_FILE}")
endif()