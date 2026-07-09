cmake_minimum_required(VERSION 3.20)
project(skainet_kernels C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(SKAINET_KERNEL_SOURCES
    src/skainet_smoke.c
    src/q4k_matmul.c
    src/q5k_matmul.c
    src/q6k_matmul.c
    src/fp32_matmul.c
    src/bf16_matmul.c
    src/q8_0_matmul.c
    src/q4_0_matmul.c
)

# SHARED: consumed by the JVM via java.lang.foreign (FFM), bundled as a JAR
#         resource (libskainet_kernels.{so,dylib,dll}).
# STATIC: consumed by Kotlin/Native via cinterop, linked into the K/N binary
#         (libskainet_kernels.a). Same sources + flags; both keep -fPIC.
add_library(skainet_kernels SHARED ${SKAINET_KERNEL_SOURCES})
add_library(skainet_kernels_static STATIC ${SKAINET_KERNEL_SOURCES})
set_target_properties(skainet_kernels_static PROPERTIES OUTPUT_NAME skainet_kernels)

set(SKAINET_KERNEL_TARGETS skainet_kernels skainet_kernels_static)

foreach(tgt IN LISTS SKAINET_KERNEL_TARGETS)
    target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Strip the "lib" prefix on Windows so the shared artifact name matches
    # the resource-bundle path skainet_kernels.{dll,so,dylib}.
    if(WIN32)
        set_target_properties(${tgt} PROPERTIES PREFIX "")
    endif()

    # Per-compiler tuning. The matmul hot loops are straight-line FP arithmetic
    # that auto-vectorizes under aggressive optimization (AVX2 on x86_64, NEON
    # on ARM64). Visibility is handled here on ELF / Mach-O; on Windows the
    # SKAINET_API macro adds dllexport.
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${tgt} PRIVATE
            -fvisibility=hidden
            -Wall -Wextra
            -O3
            -ffast-math
            -funroll-loops
        )
        # AArch64: enable the NEON paths guarded by __ARM_NEON in skainet_simd.h.
        # The SL2610 is Cortex-A55-class (ARMv8.2-A) — it HAS NEON + fp16 +
        # dotprod (asimddp) but NOT i8mm (that is ARMv8.6). Do NOT add +i8mm: it
        # would SIGILL on the board. CONFIRM on-device first: `grep Features
        # /proc/cpuinfo` must list `asimddp` and `fphp`/`asimdhp`. Explicit
        # -march (not -mcpu=native) because this is a cross-compile from x86.
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            target_compile_options(${tgt} PRIVATE
                -march=armv8.2-a+fp16+dotprod
            )
        endif()
        set_target_properties(${tgt} PROPERTIES C_VISIBILITY_PRESET hidden)
    elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${tgt} PRIVATE
            /O2
            /fp:fast
            /W3
        )
    endif()
endforeach()
