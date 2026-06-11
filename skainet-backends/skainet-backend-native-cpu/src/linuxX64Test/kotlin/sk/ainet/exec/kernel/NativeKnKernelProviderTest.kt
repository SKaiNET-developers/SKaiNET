package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelRegistry

/**
 * Verifies the K/N kernel provider integrates with [KernelRegistry] the way the
 * eager runtime's `DefaultCpuOps.chooseQuantizedMatmulHeap` resolves kernels:
 * after [installNativeKernels], the highest-priority available provider is the
 * cinterop one, and its Q5_K kernel is the registry-resolved kernel that runs.
 */
class NativeKnKernelProviderTest {

    @BeforeTest
    fun clean() = KernelRegistry.clearForTesting()

    @AfterTest
    fun reset() = KernelRegistry.clearForTesting()

    @Test
    fun installs_and_resolves_native_quant_kernels() {
        installNativeKernels()

        // Priority 100 cinterop beats the scalar (0) fallback.
        assertEquals("native-cinterop", KernelRegistry.bestAvailable()?.name)

        val provider = KernelRegistry.providers().firstOrNull { it.isAvailable() && it.matmulQ5K() != null }
        assertNotNull(provider, "no available provider carries a Q5_K kernel")
        assertSame(NativeKnQ5KMatmulKernel, provider.matmulQ5K())
        assertTrue(provider.supports("matmul", listOf("Float32", "Q5_K")))
    }

    @Test
    fun registry_resolved_q5k_kernel_is_correct() {
        installNativeKernels()
        val kernel = KernelRegistry.bestAvailable()!!.matmulQ5K()!!

        val inputDim = 1024
        val outputDim = 64
        val numBlocks = (inputDim / 256) * outputDim
        val packed = ByteArray(numBlocks * 176).also { Random(5).nextBytes(it) }
        for (b in 0 until numBlocks) {
            val base = b * 176
            packed[base] = 0x00; packed[base + 1] = 0x3C // d = 1.0f16
            packed[base + 2] = 0x00; packed[base + 3] = 0x3C // dMin = 1.0f16
        }
        val input = FloatArray(inputDim) { Random(it + 1).nextFloat() - 0.5f }

        val ref = FloatArray(outputDim)
        ScalarQ5_KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, ref, 0)
        val got = FloatArray(outputDim)
        kernel.matmul(input, 0, packed, 0, inputDim, outputDim, got, 0)

        for (o in 0 until outputDim) {
            val diff = abs(ref[o] - got[o])
            assertTrue(diff <= 5e-2f || diff / (abs(ref[o]) + 1e-9f) < 1e-4f, "row $o: ref=${ref[o]} got=${got[o]}")
        }
    }
}
