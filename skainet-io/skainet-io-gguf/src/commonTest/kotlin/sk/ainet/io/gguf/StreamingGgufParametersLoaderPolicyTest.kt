package sk.ainet.io.gguf

import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8

/**
 * Unit tests for `StreamingGgufParametersLoader.validatePolicy` — the
 * eager check inside the `withPolicy` factory that fails fast when
 * a requested [DTypePolicy] can never be satisfied by the loader's
 * current capabilities. Mirrors the SafeTensors policy-adapter test
 * (W0b) for the GGUF side (W0c of #615).
 */
class StreamingGgufParametersLoaderPolicyTest {

    @Test
    fun any_passes_validation() {
        // No throw.
        StreamingGgufParametersLoader.validatePolicy(DTypePolicy.Any)
    }

    @Test
    fun require_fp32_passes_validation() {
        StreamingGgufParametersLoader.validatePolicy(DTypePolicy.Require(FP32))
    }

    @Test
    fun require_bf16_fails_fast_with_clear_message() {
        val ex = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader.validatePolicy(DTypePolicy.Require(BF16))
        }
        val msg = ex.message ?: ""
        assertTrue(msg.contains("Require(BF16)"), msg)
        assertTrue(msg.contains("KEEP_NATIVE"), msg)
    }

    @Test
    fun require_fp16_fails_fast_with_clear_message() {
        val ex = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader.validatePolicy(DTypePolicy.Require(FP16))
        }
        assertTrue(ex.message?.contains("Require(FP16)") == true, ex.message ?: "")
    }

    @Test
    fun require_unsupported_target_fails_fast() {
        val ex = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader.validatePolicy(DTypePolicy.Require(Int8))
        }
        val msg = ex.message ?: ""
        assertTrue(msg.contains("Require(Int8)"), msg)
        assertTrue(msg.contains("does not cast"), msg)
    }

    @Test
    fun prefer_and_oneOf_always_pass_validation() {
        // Soft policies fall through silently in the loader, so the
        // validator must let them all through regardless of target.
        StreamingGgufParametersLoader.validatePolicy(DTypePolicy.Prefer(BF16))
        StreamingGgufParametersLoader.validatePolicy(DTypePolicy.Prefer(FP16))
        StreamingGgufParametersLoader.validatePolicy(DTypePolicy.Prefer(Int8))
        StreamingGgufParametersLoader.validatePolicy(DTypePolicy.OneOf(setOf(FP32, BF16)))
        StreamingGgufParametersLoader.validatePolicy(DTypePolicy.OneOf(setOf(FP16)))
    }
}
