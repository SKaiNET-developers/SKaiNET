package sk.ainet.exec.kernel

import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelStrictness
import sk.ainet.backend.api.kernel.NoSuchKernelException
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32

/**
 * Tests the [KernelStrictness] fail-fast affordance and verifies
 * that the happy path (FP32 × FP32 matmul, which always finds an
 * SPI kernel) does NOT throw when strict mode is enabled — the
 * exception is only raised when dispatch genuinely can't resolve a
 * kernel.
 */
class KernelStrictnessTest {

    private var savedProperty: String? = null

    @BeforeTest
    fun saveProperty() {
        savedProperty = System.getProperty(KernelStrictness.SYSTEM_PROPERTY)
    }

    @AfterTest
    fun restoreProperty() {
        val saved = savedProperty
        if (saved != null) {
            System.setProperty(KernelStrictness.SYSTEM_PROPERTY, saved)
        } else {
            System.clearProperty(KernelStrictness.SYSTEM_PROPERTY)
        }
    }

    @Test
    fun isEnabled_reads_system_property() {
        System.clearProperty(KernelStrictness.SYSTEM_PROPERTY)
        assertEquals(false, KernelStrictness.isEnabled(), "default must be off")

        System.setProperty(KernelStrictness.SYSTEM_PROPERTY, "true")
        assertEquals(true, KernelStrictness.isEnabled(), "property = 'true' enables strict mode")

        System.setProperty(KernelStrictness.SYSTEM_PROPERTY, "false")
        assertEquals(false, KernelStrictness.isEnabled(), "property = 'false' disables strict mode")

        System.setProperty(KernelStrictness.SYSTEM_PROPERTY, "yes")
        assertEquals(false, KernelStrictness.isEnabled(), "only 'true' enables; other values are off")
    }

    @Test
    fun failIfStrict_is_noop_when_disabled() {
        System.clearProperty(KernelStrictness.SYSTEM_PROPERTY)
        var called = false
        KernelStrictness.failIfStrict {
            called = true
            "should never be evaluated"
        }
        assertEquals(false, called, "message lambda must not run when strict is off")
    }

    @Test
    fun failIfStrict_throws_when_enabled() {
        System.setProperty(KernelStrictness.SYSTEM_PROPERTY, "true")
        val ex = assertFailsWith<NoSuchKernelException> {
            KernelStrictness.failIfStrict { "matmul (FP32 × Q4_K) has no SPI kernel" }
        }
        assertTrue(
            ex.message?.contains("FP32 × Q4_K") == true,
            "exception message must come from the lambda: '${ex.message}'",
        )
    }

    @Test
    fun fp32_matmul_does_not_throw_under_strict_mode() {
        // The FP32 path always resolves via fp32MatmulKernel (falls back
        // to ScalarMatmulKernel if registry is empty). Strict mode must
        // NOT break the happy path.
        System.setProperty(KernelStrictness.SYSTEM_PROPERTY, "true")
        val ctx = DirectCpuExecutionContext.create()
        val a = tensor<FP32, Float>(ctx, FP32::class) {
            tensor { shape(2, 3) { from(1f, 2f, 3f, 4f, 5f, 6f) } }
        }
        val b = tensor<FP32, Float>(ctx, FP32::class) {
            tensor { shape(3, 2) { from(1f, 2f, 3f, 4f, 5f, 6f) } }
        }
        // No throw expected — FP32 matmul has a resolved kernel.
        val c = ctx.ops.matmul(a, b)
        assertEquals(Shape(2, 2), c.data.shape, "happy-path FP32 matmul must work under strict mode")
    }
}
