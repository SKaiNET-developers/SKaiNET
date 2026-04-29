cmake_minimum_required(VERSION 3.20)
project(skainet_kernels C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(skainet_kernels SHARED
    src/skainet_smoke.c
)

target_include_directories(skainet_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Strip the "lib" prefix on Windows so the artifact name is consistent
# with the resource-bundle path skainet_kernels.{dll,so,dylib}.
if(WIN32)
    set_target_properties(skainet_kernels PROPERTIES PREFIX "")
endif()

# Hide non-exported symbols on ELF / Mach-O for a smaller surface area.
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(skainet_kernels PRIVATE -fvisibility=hidden -Wall -Wextra)
    set_target_properties(skainet_kernels PROPERTIES C_VISIBILITY_PRESET hidden)
endif()
