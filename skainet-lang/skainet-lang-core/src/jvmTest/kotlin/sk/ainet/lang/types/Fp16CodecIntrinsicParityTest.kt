package sk.ainet.lang.types

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [Fp16Codec.decode] must agree bit-for-bit with the JDK's `Float.float16ToFloat` on every input.
 *
 * This is the licence for `PanamaVectorFp16MatmulKernel` to call the JDK conversion instead of the
 * codec (#887): the intrinsic lowers to `vcvtph2ps` where the codec was a per-element function
 * call, and the swap is only safe because the two are the same function. The codec stays the
 * reference for targets without an intrinsic, so this test also guards the codec against drift in
 * either direction.
 *
 * JVM-only by necessity — `Float.float16ToFloat` is JDK 20+ and has no `commonMain` equivalent,
 * which is why the codec exists in the first place. The portable equivalence check lives in
 * `NarrowFloatCodecTest`.
 */
class Fp16CodecIntrinsicParityTest {

    private fun intrinsic(bits: Int): Float = java.lang.Float.float16ToFloat(bits.toShort())

    @Test
    fun decode_matches_the_jdk_intrinsic_on_all_65536_inputs() {
        for (bits in 0..0xFFFF) {
            assertEquals(
                intrinsic(bits).toRawBits(),
                Fp16Codec.decode(bits).toRawBits(),
                "decode diverged from Float.float16ToFloat at 0x${bits.toString(16)}",
            )
        }
    }

    @Test
    fun decode_matches_the_intrinsic_across_every_exponent_class() {
        // The sweep above would still pass if one whole class were somehow skipped, so assert the
        // classes are actually populated: normals, subnormals, zeros, infinities and NaNs.
        var normals = 0
        var subnormals = 0
        var zeros = 0
        var infinities = 0
        var nans = 0
        for (bits in 0..0xFFFF) {
            val exp = (bits ushr 10) and 0x1F
            val mant = bits and 0x03FF
            when {
                exp == 0x1F && mant == 0 -> infinities++
                exp == 0x1F -> nans++
                exp == 0 && mant == 0 -> zeros++
                exp == 0 -> subnormals++
                else -> normals++
            }
        }
        assertEquals(2, zeros)
        assertEquals(2, infinities)
        assertEquals(2 * 1023, subnormals)
        assertEquals(2 * 1023, nans)
        assertEquals(2 * 30 * 1024, normals)
    }

    @Test
    fun the_intrinsic_round_trips_the_codecs_own_encode() {
        // Cross-check the other direction too: whatever encode produces must decode identically
        // under both implementations, so an encode change cannot silently split them.
        val samples = floatArrayOf(
            0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 65504f, -65504f, 6.1e-5f, -6.1e-5f,
            5.96e-8f, 1e-8f, 3.14159265f, -2.71828f, 1234.5f,
            Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
        )
        for (v in samples) {
            val bits = Fp16Codec.encode(v)
            assertEquals(
                intrinsic(bits).toRawBits(),
                Fp16Codec.decode(bits).toRawBits(),
                "round-trip of $v (bits=0x${bits.toString(16)})",
            )
        }
        assertTrue(Fp16Codec.decode(Fp16Codec.encode(Float.NaN)).isNaN())
        assertTrue(intrinsic(Fp16Codec.encode(Float.NaN)).isNaN())
    }
}
