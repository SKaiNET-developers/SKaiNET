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
    src/q4k_matmul.c
    src/fp32_matmul.c
    src/bf16_matmul.c
    src/q8_0_matmul.c
)

target_include_directories(skainet_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Strip the "lib" prefix on Windows so the artifact name is consistent
# with the resource-bundle path skainet_kernels.{dll,so,dylib}.
if(WIN32)
    set_target_properties(skainet_kernels PROPERTIES PREFIX "")
endif()

# Per-compiler tuning. The Q4_K kernel hot loop is straight-line FP
# arithmetic that auto-vectorizes cleanly under aggressive optimization
# (AVX2 on x86_64, NEON on ARM64). Visibility is also handled here on
# ELF / Mach-O; on Windows the SKAINET_API macro adds dllexport so we
# don't need /VISIBILITY flags.
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(skainet_kernels PRIVATE
        -fvisibility=hidden
        -Wall -Wextra
        -O3
        -ffast-math
        -funroll-loops
    )
    set_target_properties(skainet_kernels PROPERTIES C_VISIBILITY_PRESET hidden)
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    target_compile_options(skainet_kernels PRIVATE
        /O2
        /fp:fast
        /W3
    )
endif()
