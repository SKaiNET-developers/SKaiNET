package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.KernelRegistry
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class NativeFfmPipelineTest {

    @AfterTest
    fun resetRegistry() {
        KernelRegistry.clearForTesting()
    }

    @Test
    fun `smoke kernel doubles its inputs end-to-end via FFM`() {
        assertTrue(
            NativeLibraryLoader.isLoaded(),
            "Bundled libskainet_kernels resource missing for the host platform — " +
                "did the CMake build run before jvmProcessResources?",
        )
        assertTrue(NativeFfmSmoke.isAvailable(), "skainet_smoke_double symbol not resolved")

        val input = floatArrayOf(0f, 1f, -2.5f, 3.14159f, 1e6f)
        val output = NativeFfmSmoke.double(input)
        assertNotNull(output)
        for (i in input.indices) {
            assertEquals(2.0f * input[i], output[i], 0f, "index $i")
        }
    }

    @Test
    fun `provider exposes Q4_K and FP32 kernels when the native lib loads`() {
        assertEquals("native-ffm", NativeKernelProvider.name)
        assertEquals(100, NativeKernelProvider.priority)
        assertTrue(
            NativeKernelProvider.isAvailable(),
            "Native kernel provider reports unavailable on this host — " +
                "bundled libskainet_kernels missing or skainet_q4k_matmul unresolved",
        )
        // PR 5 wires both Q4_K (PR 2) and FP32 (this PR) through the SPI.
        assertNotNull(NativeKernelProvider.matmulFp32())
        assertNotNull(NativeKernelProvider.matmulQ4K())
    }

    @Test
    fun `factory delegates to the singleton`() {
        val factory = NativeKernelProviderFactory()
        assertEquals(NativeKernelProvider.name, factory.name)
        assertEquals(NativeKernelProvider.priority, factory.priority)
        assertEquals(NativeKernelProvider.isAvailable(), factory.isAvailable())
    }
}
