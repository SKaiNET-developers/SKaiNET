package sk.ainet.bench.publish.env

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelServiceLoader
import sk.ainet.bench.publish.schema.RuntimeInfo
import java.lang.management.ManagementFactory

public object RuntimeInfoProvider {

    public fun collect(selectedProvider: String): RuntimeInfo {
        ensureProvidersInstalled()
        val available = KernelRegistry.availableNames()
        return RuntimeInfo(
            name = "skainet-engine",
            variant = "upstream",
            version = readVersion(),
            commit = readCommit(),
            backend = "cpu",
            kernelProvider = selectedProvider,
            availableProviders = available,
        )
    }

    private fun ensureProvidersInstalled() {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
    }

    private fun readVersion(): String {
        val res = RuntimeInfoProvider::class.java.classLoader
            .getResourceAsStream("skainet-engine-publish.properties")
        if (res != null) {
            val props = java.util.Properties()
            res.use(props::load)
            val v = props.getProperty("version")
            if (!v.isNullOrBlank()) return v
        }
        return System.getProperty("skainet.version", "unspecified")
    }

    private fun readCommit(): String {
        val res = RuntimeInfoProvider::class.java.classLoader
            .getResourceAsStream("skainet-engine-publish.properties")
        if (res != null) {
            val props = java.util.Properties()
            res.use(props::load)
            val c = props.getProperty("commit")
            if (!c.isNullOrBlank()) return c
        }
        return System.getProperty("skainet.commit", "unspecified")
    }

    public fun jvmArgs(): List<String> = ManagementFactory.getRuntimeMXBean().inputArguments
}
