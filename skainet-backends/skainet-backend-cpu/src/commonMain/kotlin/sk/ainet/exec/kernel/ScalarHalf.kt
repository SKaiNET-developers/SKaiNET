package sk.ainet.exec.kernel

/**
 * Convert a 16-bit IEEE-754 half-precision value (low 16 bits of [hbits])
 * to FP32. Shared by the scalar packed-quant kernels in this package
 * (Q5_1/Q5_0/Q4_K/Q6_K). Mirrors the inlined helpers in
 * [ScalarQ4_0MatmulKernel] / [ScalarQ8_0MatmulKernel].
 */
internal fun decodeHalf(hbits: Int): Float {
    val sign = (hbits and 0x8000) shl 16
    val exp = (hbits and 0x7C00) shr 10
    val mant = hbits and 0x03FF
    return when (exp) {
        0 -> {
            if (mant == 0) {
                Float.fromBits(sign)
            } else {
                var m = mant
                var e = -14
                while ((m and 0x400) == 0) {
                    m = m shl 1
                    e--
                }
                m = m and 0x3FF
                Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
            }
        }
        31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
        else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
    }
}
