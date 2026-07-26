package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `convert` to a 16-bit float target must actually round.
 *
 * It previously shared the FP32 branch, so `convert(x, FP16)` only re-tagged the dtype while
 * keeping full FP32 precision — the tensor claimed to be FP16 while holding values no FP16 can
 * represent, with no rounding, no overflow to Inf past the format's range, and no NaN handling.
 * BF16 was rejected as a target outright.
 */
class NarrowFloatConvertTest {

    private val ctx = DirectCpuExecutionContext()

    private fun fp32(vararg values: Float) =
        ctx.fromFloatArray<FP32, Float>(Shape(values.size), FP32::class, values)

    @Test
    fun convert_to_fp16_rounds_values_to_binary16() {
        // 1 + 2^-20 is far below binary16's 2^-10 step, so it must collapse onto 1.0.
        val src = fp32(1.0f + 1.0f / 1048576.0f, 3.14159265f, -2.71828f)
        val out = ctx.ops.convert(src, FP16).data.copyToFloatArray()

        assertEquals(1.0f, out[0], "sub-ulp detail must be rounded away, not retained")
        for ((i, v) in floatArrayOf(1.0f + 1.0f / 1048576.0f, 3.14159265f, -2.71828f).withIndex()) {
            assertEquals(
                Fp16Codec.decode(Fp16Codec.encode(v)), out[i],
                "element $i must equal the binary16 rounding of the input",
            )
        }
    }

    @Test
    fun convert_to_bf16_is_supported_and_truncates() {
        val values = floatArrayOf(1.0f, 3.14159265f, -2.71828f, 1234.5f)
        val out = ctx.ops.convert(fp32(*values), BF16).data.copyToFloatArray()
        for ((i, v) in values.withIndex()) {
            assertEquals(
                Bf16Codec.decode(Bf16Codec.encode(v)), out[i],
                "element $i must equal the bf16 encoding of the input",
            )
        }
    }

    @Test
    fun convert_to_fp16_overflows_to_infinity_past_the_ceiling() {
        // The old retag path kept 70000f verbatim in a tensor claiming to be FP16, a value
        // binary16 cannot represent at all.
        val out = ctx.ops.convert(fp32(70000f, -70000f), FP16).data.copyToFloatArray()
        assertTrue(out[0].isInfinite() && out[0] > 0, "must overflow to +Inf, got ${out[0]}")
        assertTrue(out[1].isInfinite() && out[1] < 0, "must overflow to -Inf, got ${out[1]}")
    }

    @Test
    fun convert_to_bf16_keeps_the_same_value_finite() {
        // bf16 has FP32's exponent range, so the same input that overflows fp16 stays finite.
        val out = ctx.ops.convert(fp32(70000f), BF16).data.copyToFloatArray()
        assertTrue(out[0].isFinite(), "bf16 must not overflow at 70000, got ${out[0]}")
    }

    @Test
    fun convert_to_fp16_flushes_subnormal_underflow_to_zero() {
        // 2^-30 is far below binary16's smallest subnormal (2^-24).
        val out = ctx.ops.convert(fp32(1.0f / 1073741824.0f), FP16).data.copyToFloatArray()
        assertEquals(0.0f, out[0], "must underflow to zero rather than retain the FP32 value")
    }

    @Test
    fun convert_reports_the_target_dtype() {
        assertEquals(FP16::class, ctx.ops.convert(fp32(1f), FP16).dtype)
        assertEquals(BF16::class, ctx.ops.convert(fp32(1f), BF16).dtype)
    }

    @Test
    fun convert_to_fp32_is_unchanged() {
        val values = floatArrayOf(1.0f, 3.14159265f, -2.71828f)
        val out = ctx.ops.convert(fp32(*values), FP32).data.copyToFloatArray()
        for ((i, v) in values.withIndex()) assertEquals(v, out[i], "FP32 target must be exact")
    }

    @Test
    fun round_tripping_through_fp16_is_idempotent() {
        // Rounding an already-rounded value must not move it again.
        val src = fp32(3.14159265f, -2.71828f, 1234.5f)
        val once = ctx.ops.convert(src, FP16).data.copyToFloatArray()
        val twice = ctx.ops.convert(
            ctx.fromFloatArray<FP32, Float>(Shape(once.size), FP32::class, once), FP16,
        ).data.copyToFloatArray()
        for (i in once.indices) assertEquals(once[i], twice[i], "element $i must be stable")
    }
}
