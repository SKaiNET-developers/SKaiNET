package sk.ainet.io.safetensors

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32

/**
 * Verifies the [Bf16LoadPolicy.toDTypePolicy] adapter — the bridge
 * between the BF16-specific enum (existing prior art) and the
 * generalised [DTypePolicy] sealed type (W1 of #615). Confirms
 * both arms of the enum land on equivalent `Require` policies so
 * downstream code paths can flow through `DTypePolicy` uniformly.
 */
class Bf16LoadPolicyToDTypePolicyTest {

    @Test
    fun dequant_to_fp32_maps_to_require_fp32() {
        val policy = Bf16LoadPolicy.DEQUANT_TO_FP32.toDTypePolicy()
        assertEquals(DTypePolicy.Require(FP32), policy)
    }

    @Test
    fun keep_native_maps_to_require_bf16() {
        val policy = Bf16LoadPolicy.KEEP_NATIVE.toDTypePolicy()
        assertEquals(DTypePolicy.Require(BF16), policy)
    }

    @Test
    fun adapter_covers_every_enum_arm() {
        // Defensive: if a new arm is added to Bf16LoadPolicy without
        // also updating toDTypePolicy, this test surfaces it because
        // toDTypePolicy's `when` is exhaustive — Kotlin emits a
        // compile error rather than silently dropping the case.
        for (arm in Bf16LoadPolicy.entries) {
            val mapped = arm.toDTypePolicy()
            assertEquals(true, mapped is DTypePolicy.Require, "$arm must map to Require, got $mapped")
        }
    }
}
