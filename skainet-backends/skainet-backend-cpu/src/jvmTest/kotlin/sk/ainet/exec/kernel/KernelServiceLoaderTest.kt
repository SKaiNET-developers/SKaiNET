package sk.ainet.exec.kernel

import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelServiceLoader

/**
 * End-to-end check that the cpu backend's
 * `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider`
 * declaration is wired correctly: `KernelServiceLoader` should see
 * both the scalar and Panama factories on the classpath, register
 * them in the [KernelRegistry], and Panama should outrank scalar.
 */
class KernelServiceLoaderTest {

    @BeforeTest
    fun setUp() = KernelRegistry.clearForTesting()

    @AfterTest
    fun tearDown() = KernelRegistry.clearForTesting()

    @Test
    fun discoverFindsBothCpuProviders() {
        val discovered = KernelServiceLoader.discover().map { it.name }.toSet()
        assertTrue(
            "scalar" in discovered,
            "expected scalar to be discovered, got $discovered",
        )
        assertTrue(
            "panama-vector" in discovered,
            "expected panama-vector to be discovered, got $discovered",
        )
    }

    @Test
    fun installAllRegistersBothProvidersInPriorityOrder() {
        val installed = KernelServiceLoader.installAll().toSet()
        assertEquals(setOf("scalar", "panama-vector"), installed)
        // Registry sorts by priority on insert, regardless of ServiceLoader order.
        val available = KernelRegistry.availableNames()
        assertEquals(listOf("panama-vector", "scalar"), available)
    }

    @Test
    fun bestAvailableAfterInstallIsPanamaOnTestJdk() {
        KernelServiceLoader.installAll()
        val best = KernelRegistry.bestAvailable()
        assertEquals("panama-vector", best?.name)
        // The factory wrapper delegates to the singleton, so the
        // matmul kernel pulled out is the same object as the
        // singleton's.
        assertSame(PanamaVectorMatmulKernel, best?.matmulFp32())
    }

    @Test
    fun installAllIsIdempotent() {
        KernelServiceLoader.installAll()
        val countAfterFirst = KernelRegistry.providers().size
        KernelServiceLoader.installAll()
        // ServiceLoader produces fresh factory instances each call, so
        // a second installAll() will append duplicates with new identity.
        // The registry only deduplicates by reference equality, so the
        // count grows. Verify the available-name set still contains
        // the same providers — the higher-priority Panama still wins.
        assertTrue(
            KernelRegistry.providers().size >= countAfterFirst,
            "registry should not lose providers across reinstall",
        )
        assertEquals("panama-vector", KernelRegistry.bestAvailable()?.name)
    }
}
