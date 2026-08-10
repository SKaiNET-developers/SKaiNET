package sk.ainet.exec.kernel.jni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Host-JVM behavior contract: on a machine without the Android `.so`
 * (every host unit-test run), the provider must degrade gracefully —
 * unavailable, all kernels null, and absolutely no exception escaping.
 * This mirrors how `NativeKernelProvider` behaves on JVMs where the FFM
 * library doesn't load, and it is what the registry's cascade relies on.
 */
class JniKernelProviderHostTest {

    @Test
    fun unavailable_without_native_library_and_never_throws() {
        assertFalse(JniKernelProvider.isAvailable())
        assertNull(JniKernelProvider.activeVariant)
        assertNull(JniKernelProvider.matmulQ8_0())
        assertNull(JniKernelProvider.matmulQ4_0())
        assertNull(JniKernelProvider.matmulQ4K())
        assertNull(JniKernelProvider.matmulQ5K())
        assertNull(JniKernelProvider.matmulQ6K())
        assertNull(JniKernelProvider.matmulFp32())
    }

    @Test
    fun provider_contract_name_and_priority() {
        assertEquals("native-jni", JniKernelProvider.name)
        assertEquals(100, JniKernelProvider.priority)
    }

    @Test
    fun supports_reports_false_when_unavailable() {
        assertFalse(JniKernelProvider.supports("matmul", listOf("Float32", "Q8_0")))
        assertFalse(JniKernelProvider.supports("matmul", listOf("Float32", "Q4_K")))
    }

    @Test
    fun serviceloader_factory_delegates() {
        val factory = JniKernelProviderFactory()
        assertEquals("native-jni", factory.name)
        assertEquals(100, factory.priority)
        assertFalse(factory.isAvailable())
    }
}
