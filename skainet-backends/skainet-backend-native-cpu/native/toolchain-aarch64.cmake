# CMake toolchain for cross-compiling the native kernels to aarch64 Linux
# from an x86_64 host (e.g. the SL2610 board build / CI). Requires the
# `gcc-aarch64-linux-gnu` package (or an equivalent clang cross setup).
#
# Usage:
#   cmake -S native -B build/native/cmake-build-arm64 \
#         -DCMAKE_TOOLCHAIN_FILE=native/toolchain-aarch64.cmake \
#         -DCMAKE_BUILD_TYPE=Release
#
# CMAKE_SYSTEM_PROCESSOR=aarch64 makes CMakeLists.txt take the
# `-march=armv8.2-a+fp16+dotprod` branch that enables the __ARM_NEON paths.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Allow overriding the cross compiler (e.g. a clang cross or a different
# triple) via -DSKAINET_AARCH64_CC=... ; default to the Debian/Ubuntu GNU
# cross toolchain.
if(NOT DEFINED SKAINET_AARCH64_CC)
    set(SKAINET_AARCH64_CC aarch64-linux-gnu-gcc)
endif()
set(CMAKE_C_COMPILER ${SKAINET_AARCH64_CC})

# Search for libraries/headers only in the target sysroot, but find
# programs (the compiler) on the host.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
