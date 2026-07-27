package sk.ainet.io.safetensors

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8

/**
 * Unit tests for the `DTypePolicy` → `Bf16LoadPolicy` adapter in
 * [SafeTensorsParametersLoader.mapPolicyToBf16]. The `withPolicy`
 * factory is a thin wrapper over this mapper plus the existing
 * constructor; testing the mapper covers the routing logic without
 * needing a real SafeTensors fixture.
 */
class SafeTensorsParametersLoaderPolicyTest {

    @Test
    fun any_maps_to_dequant_to_fp32() {
        assertEquals(
            Bf16LoadPolicy.DEQUANT_TO_FP32,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Any),
        )
    }

    @Test
    fun require_bf16_maps_to_keep_native() {
        assertEquals(
            Bf16LoadPolicy.KEEP_NATIVE,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Require(BF16)),
        )
    }

    @Test
    fun require_fp32_maps_to_dequant() {
        assertEquals(
            Bf16LoadPolicy.DEQUANT_TO_FP32,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Require(FP32)),
        )
    }

    @Test
    fun require_fp16_now_keeps_f16_sources_native() {
        // Previously this threw: there was no Fp16DenseTensorData to back a KEEP_NATIVE F16 path.
        // That backing now exists, so Require(FP16) is satisfiable for F16 sources.
        assertEquals(
            NarrowFloatLoadPolicy.KEEP_NATIVE,
            SafeTensorsParametersLoader.mapPolicyToFp16(DTypePolicy.Require(FP16)),
        )
    }

    @Test
    fun the_two_narrow_formats_do_not_keep_each_other_native() {
        // Neither format can be produced from the other without a lossy re-encode, so a policy
        // naming one must widen the other rather than silently mis-tagging it.
        assertEquals(
            NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
            SafeTensorsParametersLoader.mapPolicyToFp16(DTypePolicy.Require(BF16)),
            "Require(BF16) must not keep F16 sources packed",
        )
        assertEquals(
            NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Require(FP16)),
            "Require(FP16) must not keep BF16 sources packed",
        )
    }

    @Test
    fun require_fp16_still_keeps_bf16_behaviour_intact() {
        assertEquals(
            NarrowFloatLoadPolicy.KEEP_NATIVE,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Require(BF16)),
        )
        assertEquals(
            NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Require(FP32)),
        )
    }

    @Test
    fun require_unsupported_target_fails_with_explicit_message() {
        val ex = assertFailsWith<IllegalArgumentException> {
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Require(Int8))
        }
        val msg = ex.message ?: ""
        assertEquals(true, msg.contains("Require(Int8)"), "msg: $msg")
        assertEquals(true, msg.contains("cannot fabricate"), "msg: $msg")
    }

    @Test
    fun prefer_bf16_maps_to_keep_native() {
        assertEquals(
            Bf16LoadPolicy.KEEP_NATIVE,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Prefer(BF16)),
        )
    }

    @Test
    fun prefer_fp32_or_anything_else_maps_to_dequant() {
        assertEquals(
            Bf16LoadPolicy.DEQUANT_TO_FP32,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Prefer(FP32)),
        )
        assertEquals(
            Bf16LoadPolicy.DEQUANT_TO_FP32,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.Prefer(FP16)),
            "Prefer is soft — unsatisfiable preferences fall through silently, no throw",
        )
    }

    @Test
    fun oneOf_with_bf16_maps_to_keep_native() {
        assertEquals(
            Bf16LoadPolicy.KEEP_NATIVE,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.OneOf(setOf(BF16, FP32))),
        )
    }

    @Test
    fun oneOf_without_bf16_maps_to_dequant() {
        assertEquals(
            Bf16LoadPolicy.DEQUANT_TO_FP32,
            SafeTensorsParametersLoader.mapPolicyToBf16(DTypePolicy.OneOf(setOf(FP32, FP16))),
        )
    }

    @Test
    fun parity_with_bf16LoadPolicy_toDTypePolicy() {
        // Round-trip property: the BF16 enum's adapter should land on a
        // policy that the inverse mapper sends back to the original enum.
        for (arm in Bf16LoadPolicy.entries) {
            val asDTypePolicy = arm.toDTypePolicy()
            val back = SafeTensorsParametersLoader.mapPolicyToBf16(asDTypePolicy)
            assertEquals(arm, back, "round-trip failed for $arm via $asDTypePolicy")
        }
    }
}
