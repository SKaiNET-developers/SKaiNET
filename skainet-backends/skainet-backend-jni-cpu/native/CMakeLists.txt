cmake_minimum_required(VERSION 3.22)
project(skainet_jni C)

# The C kernel sources are shared with skainet-backend-native-cpu — this
# module adds only the JNI shim. Single source of truth for the math.
set(SKAINET_KERNELS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../skainet-backend-native-cpu/native)

set(SKAINET_KERNEL_SOURCES
    ${SKAINET_KERNELS_ROOT}/src/skainet_smoke.c
    ${SKAINET_KERNELS_ROOT}/src/fp32_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/bf16_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/fp16_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/q4_0_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/q5_0_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/q5_1_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/q8_0_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/skainet_row_threads.c
    ${SKAINET_KERNELS_ROOT}/src/q4k_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/q5k_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/q6k_matmul.c
    ${SKAINET_KERNELS_ROOT}/src/bitnet_gemv.c
    ${SKAINET_KERNELS_ROOT}/src/skainet_ternary_f32.c
    # Vendored NeoGPU ternary LUT kernel (MIT, verbatim). NDK toolchains are
    # clang, and bionic bundles pthreads — no Threads link needed.
    ${SKAINET_KERNELS_ROOT}/src/vendor/neogpu/hs_ml_ternary_neon.c
)

set(SKAINET_JNI_SHIM ${CMAKE_CURRENT_SOURCE_DIR}/skainet_jni.c)

# Optimization flags match the sibling CMakeLists (-O3 -ffast-math) so the
# JNI path performs identically to the FFM/cinterop consumers of the same C.
set(SKAINET_C_FLAGS -O3 -ffast-math -funroll-loops)

# Android 15+ devices ship 16 KB pages; .so files must be max-page-size
# aligned. Recent NDKs default to this, the explicit flag pins it.
set(SKAINET_LINK_FLAGS -Wl,-z,max-page-size=16384)

function(skainet_add_jni_lib target)
    add_library(${target} SHARED ${SKAINET_JNI_SHIM} ${SKAINET_KERNEL_SOURCES})
    target_include_directories(${target} PRIVATE ${SKAINET_KERNELS_ROOT}/include)
    target_compile_options(${target} PRIVATE ${SKAINET_C_FLAGS})
    target_compile_definitions(${target} PRIVATE SKAINET_HAVE_NEOGPU_TERNARY)
    target_link_options(${target} PRIVATE ${SKAINET_LINK_FLAGS})
endfunction()

# Baseline: NDK default -march (armv8-a on arm64-v8a). NEON is architecturally
# guaranteed on AArch64, so fp32/q8_0/q4_0/q5k get their SIMD bodies, while the
# dotprod-guarded q4k/q6k paths fall back to scalar. Runs on EVERY arm64 core,
# including armv8.0 (Cortex-A53/A55-r0) — no SIGILL possible.
skainet_add_jni_lib(skainet_jni)

# v8.2 variant: enables the vdotq_s32 paths in q4k/q6k (and fp16 storage ops).
# Would SIGILL on armv8.0 cores — JniKernels loads it only after confirming
# `asimddp` (+ `asimdhp`/`fphp`) in /proc/cpuinfo. On non-ARM ABIs (x86_64
# emulator) the flags don't apply and the two libs are identical; both are
# still packaged so the loader's two-tier logic needs no per-ABI special case.
skainet_add_jni_lib(skainet_jni_v82)
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(skainet_jni_v82 PRIVATE -march=armv8.2-a+fp16+dotprod)
endif()
