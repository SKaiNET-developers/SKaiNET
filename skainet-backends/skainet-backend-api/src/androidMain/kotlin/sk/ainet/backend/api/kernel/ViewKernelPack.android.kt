package sk.ainet.backend.api.kernel

import java.util.ServiceLoader
import sk.ainet.lang.memory.ExperimentalMemoryApi

/**
 * Android discovery for [ViewKernelPack]. `ServiceLoader` exists on Android, so the JNI packs a
 * consumer ships (e.g. the NEON row-major pack in `skainet-backend-jni-cpu`) are discovered the
 * same way as on the JVM, provided the packaging step keeps `META-INF/services` entries.
 */
@ExperimentalMemoryApi
internal actual fun installPlatformKernelPacks(): List<String> =
    runCatching {
        ServiceLoader.load(ViewKernelPack::class.java)
            .mapNotNull { pack -> runCatching { pack.install(); pack.name }.getOrNull() }
            .toList()
    }.getOrElse { emptyList() }

@ExperimentalMemoryApi
internal actual fun installPlatformKernelProviders(): List<String> =
    runCatching { KernelServiceLoader.installAll() }.getOrElse { emptyList() }
