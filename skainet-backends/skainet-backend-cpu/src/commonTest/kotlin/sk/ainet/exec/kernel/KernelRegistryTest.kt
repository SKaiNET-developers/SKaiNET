package sk.ainet.exec.kernel

import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertSame
import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.KernelRegistry

class KernelRegistryTest {

    @BeforeTest
    fun setUp() = KernelRegistry.clearForTesting()

    @AfterTest
    fun tearDown() = KernelRegistry.clearForTesting()

    @Test
    fun emptyRegistryHasNoBest() {
        assertNull(KernelRegistry.bestAvailable())
        assertEquals(emptyList(), KernelRegistry.availableNames())
    }

    @Test
    fun scalarRegistersAndIsBest() {
        KernelRegistry.register(ScalarKernelProvider)
        assertSame(ScalarKernelProvider, KernelRegistry.bestAvailable())
        assertEquals(listOf("scalar"), KernelRegistry.availableNames())
    }

    @Test
    fun higherPriorityWins() {
        val fast = object : KernelProvider {
            override val name = "fake-fast"
            override val priority = 50
            override fun isAvailable() = true
            override fun matmulFp32(): Fp32MatmulKernel = ScalarMatmulKernel
        }
        KernelRegistry.register(ScalarKernelProvider)
        KernelRegistry.register(fast)
        assertSame(fast, KernelRegistry.bestAvailable())
    }

    @Test
    fun unavailableProviderIsSkipped() {
        val pretender = object : KernelProvider {
            override val name = "pretender"
            override val priority = 100
            override fun isAvailable() = false
            override fun matmulFp32(): Fp32MatmulKernel = ScalarMatmulKernel
        }
        KernelRegistry.register(pretender)
        KernelRegistry.register(ScalarKernelProvider)
        // pretender outranks scalar but isn't available — scalar wins.
        assertSame(ScalarKernelProvider, KernelRegistry.bestAvailable())
        // availableNames excludes pretender.
        assertEquals(listOf("scalar"), KernelRegistry.availableNames())
    }

    @Test
    fun findByNameIsCaseInsensitive() {
        KernelRegistry.register(ScalarKernelProvider)
        assertSame(ScalarKernelProvider, KernelRegistry.find("scalar"))
        assertSame(ScalarKernelProvider, KernelRegistry.find("Scalar"))
        assertSame(ScalarKernelProvider, KernelRegistry.find("SCALAR"))
        assertNull(KernelRegistry.find("unknown"))
    }

    @Test
    fun reRegisteringSameInstanceIsNoOp() {
        KernelRegistry.register(ScalarKernelProvider)
        KernelRegistry.register(ScalarKernelProvider)
        assertEquals(1, KernelRegistry.providers().size)
    }
}
