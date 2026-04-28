package sk.ainet.exec.kernel

import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelRegistry

class PanamaVectorKernelProviderTest {

    @BeforeTest
    fun setUp() = KernelRegistry.clearForTesting()

    @AfterTest
    fun tearDown() = KernelRegistry.clearForTesting()

    @Test
    fun providerHasExpectedNameAndPriority() {
        assertEquals("panama-vector", PanamaVectorKernelProvider.name)
        assertEquals(50, PanamaVectorKernelProvider.priority)
    }

    @Test
    fun isAvailableOnTestJdk() {
        // The cpu-backend test suite runs on JDK 21+ with the incubator
        // module on the module path (see jvm-cpu-jmh build script and the
        // project's JDK requirement). Vector should be available here.
        assertTrue(
            PanamaVectorKernelProvider.isAvailable(),
            "expected Panama provider to be available on the test JDK",
        )
    }

    @Test
    fun matmulFp32IsTheVectorKernelWhenAvailable() {
        assertSame(PanamaVectorMatmulKernel, PanamaVectorKernelProvider.matmulFp32())
    }

    @Test
    fun beatsScalarInRegistryWhenBothRegistered() {
        KernelRegistry.register(ScalarKernelProvider)
        KernelRegistry.register(PanamaVectorKernelProvider)
        // Higher priority wins.
        assertSame(PanamaVectorKernelProvider, KernelRegistry.bestAvailable())
        assertEquals(
            listOf("panama-vector", "scalar"),
            KernelRegistry.availableNames(),
        )
    }

    @Test
    fun killSwitchDisablesProvider() {
        val key = "skainet.cpu.vector.enabled"
        val previous = System.getProperty(key)
        try {
            System.setProperty(key, "false")
            assertEquals(false, PanamaVectorKernelProvider.isAvailable())
            assertEquals(null, PanamaVectorKernelProvider.matmulFp32())
        } finally {
            if (previous == null) System.clearProperty(key)
            else System.setProperty(key, previous)
        }
    }
}
