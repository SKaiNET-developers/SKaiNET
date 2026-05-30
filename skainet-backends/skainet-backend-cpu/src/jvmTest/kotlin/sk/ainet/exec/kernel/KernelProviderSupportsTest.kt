package sk.ainet.exec.kernel

import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelRegistry

/**
 * Capability-query parity tests for the default
 * `KernelProvider.supports(...)` implementation. The default body
 * introspects the existing per-kernel accessors and reports
 * `true` iff the accessor returns non-null; these tests confirm
 * the introspection matches what `bestAvailable()?.matmul*()`
 * actually returns for the two providers shipped today.
 */
class KernelProviderSupportsTest {

    @BeforeTest
    fun setUp() = KernelRegistry.clearForTesting()

    @AfterTest
    fun tearDown() = KernelRegistry.clearForTesting()

    @Test
    fun panama_supports_matches_accessor_nullability() {
        val p = PanamaVectorKernelProvider
        assertEquals(
            p.matmulFp32() != null,
            p.supports("matmul", listOf("Float32", "Float32")),
            "FP32 matmul support must mirror matmulFp32() != null",
        )
        assertEquals(
            p.matmulBf16() != null,
            p.supports("matmul", listOf("Float32", "BFloat16")),
            "BF16 matmul support must mirror matmulBf16() != null",
        )
        assertEquals(
            p.matmulQ4K() != null,
            p.supports("matmul", listOf("Float32", "Q4_K")),
            "Q4_K matmul support must mirror matmulQ4K() != null",
        )
        assertEquals(
            p.matmulQ8_0() != null,
            p.supports("matmul", listOf("Float32", "Q8_0")),
            "Q8_0 matmul support must mirror matmulQ8_0() != null",
        )
        assertEquals(
            p.matmulQ4_0() != null,
            p.supports("matmul", listOf("Float32", "Q4_0")),
            "Q4_0 matmul support must mirror matmulQ4_0() != null",
        )
    }

    @Test
    fun scalar_supports_matches_accessor_nullability() {
        val p = ScalarKernelProvider
        assertEquals(
            p.matmulFp32() != null,
            p.supports("matmul", listOf("Float32", "Float32")),
        )
        // Scalar declines quantized matmuls today; the capability query
        // must agree.
        assertEquals(
            p.matmulQ4K() != null,
            p.supports("matmul", listOf("Float32", "Q4_K")),
        )
        // Scalar carries the Q4_0 floor kernel, so the capability query
        // must report it as supported.
        assertTrue(p.supports("matmul", listOf("Float32", "Q4_0")))
    }

    @Test
    fun unknown_op_returns_false() {
        assertFalse(
            PanamaVectorKernelProvider.supports("sdpa", listOf("Float32", "Float32", "Float32")),
            "supports() must return false for ops the provider does not advertise",
        )
    }

    @Test
    fun matmul_with_wrong_arity_returns_false() {
        assertFalse(
            PanamaVectorKernelProvider.supports("matmul", listOf("Float32")),
            "matmul takes exactly two dtype keys (input, weight)",
        )
        assertFalse(
            PanamaVectorKernelProvider.supports("matmul", listOf("Float32", "Float32", "Float32")),
        )
    }

    @Test
    fun matmul_with_non_float32_input_returns_false() {
        // The kernel SPI today only specializes FP32-input matmuls.
        // BFloat16-input matmul is a future kernel; the capability
        // query must say "no" until that kernel exists.
        assertFalse(
            PanamaVectorKernelProvider.supports("matmul", listOf("BFloat16", "Float32")),
        )
        assertTrue(
            PanamaVectorKernelProvider.supports("matmul", listOf("Float32", "Float32")),
        )
    }
}
