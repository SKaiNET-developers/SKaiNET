#ifndef SKAINET_SIMD_H
#define SKAINET_SIMD_H

/*
 * Compile-time SIMD capability detection for the native CPU kernels.
 *
 * The kernels keep their portable scalar bodies as the `#else` fallback,
 * so x86_64 (which auto-vectorizes well under -O3 -ffast-math) and any
 * pre-ARMv8.2 target keep compiling unchanged. The NEON paths are only
 * taken when the compiler advertises `__ARM_NEON` (AArch64 always does
 * with the right -march). `__ARM_FEATURE_DOTPROD` / `__ARM_FEATURE_MATMUL_INT8`
 * are gated on the build flags (`-march=armv8.2-a+dotprod`, etc.).
 *
 * AARCH64-VERIFIED (2026-07-02): the NEON paths (fp32 vfmaq_f32, q4k
 * vdotq_s32 dotprod, q5k, q6k, q8_0) were cross-built with
 * `-march=armv8.2-a+fp16+dotprod` (aarch64 gcc 8.3, the K/N-bundled
 * toolchain) and parity-checked against the commonMain scalar references:
 *   - under qemu-aarch64 via
 *       ./gradlew :skainet-backends:skainet-backend-native-cpu:linuxArm64Test -PcrossArm64=true
 *   - AND on the physical SL2610 board (Cortex-A55, aarch64): the same
 *     test.kexe run natively on-device — 23/23 tests green, no SIGILL.
 *     `/proc/cpuinfo` confirmed asimddp + fphp/asimdhp present and i8mm
 *     absent, matching the chosen -march (no +i8mm).
 * The linked archive was confirmed to contain udot/sdot + fmla, i.e. the
 * SIMD paths — not the scalar fallback — executed. bf16 and fp16 have no
 * NEON path (scalar only).
 *
 * q4_0 gained a plain-NEON body (nibble unpack + widen + vfmaq_f32, no
 * dotprod requirement) later — parity-checked under qemu-aarch64 via the
 * same linuxArm64Test lane (#920).
 */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define SKAINET_HAVE_NEON 1
#endif

#if defined(__ARM_FEATURE_DOTPROD)
#  define SKAINET_HAVE_DOTPROD 1
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
#  define SKAINET_HAVE_I8MM 1
#endif

#ifdef SKAINET_HAVE_NEON
/* Horizontal sum of a float32x4 lane vector. AArch64 has vaddvq_f32
 * natively; this wrapper keeps call sites readable. */
static inline float skainet_neon_hadd_f32(float32x4_t v) {
    return vaddvq_f32(v);
}

/* Widen 16 unsigned bytes to four float32x4 lanes (out[0]=lanes 0..3, …). */
static inline void skainet_neon_u8x16_to_f32x4x4(uint8x16_t v, float32x4_t out[4]) {
    const uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
    const uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));
    out[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
    out[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
    out[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
    out[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));
}
#endif

#endif /* SKAINET_SIMD_H */
