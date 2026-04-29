package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.SymbolLookup
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/**
 * Locates the bundled `libskainet_kernels` shared library shipped in
 * this module's JAR resources, extracts it to a process-scoped temp
 * directory, calls `System.load` on the resulting absolute path, and
 * exposes a process-lifetime [SymbolLookup] for FFM downcalls.
 *
 * Resource layout: `native/<os>-<arch>/<libname>` where `<libname>` is
 *  - `libskainet_kernels.so` on Linux
 *  - `libskainet_kernels.dylib` on macOS
 *  - `skainet_kernels.dll` on Windows
 *
 * `<os>-<arch>` is the same tag the Gradle build uses when staging the
 * artifact (`linux-x86_64`, `linux-arm64`, `macos-arm64`, ...). PR 1
 * only ships one variant per build host; cross-arch shipping comes in
 * a later PR.
 *
 * Failure modes are non-fatal: missing resource, unsupported platform,
 * or `System.load` failure → [tryInit] returns `false` and `lookup`
 * stays `null`. Callers (notably [NativeKernelProvider.isAvailable])
 * cascade to a lower-priority provider.
 */
internal object NativeLibraryLoader {

    @Volatile
    private var initialized: Boolean = false

    @Volatile
    private var loadedLookup: SymbolLookup? = null

    /**
     * The shared [Arena] that owns the [SymbolLookup]. Lifetime is the
     * JVM process; deliberately not closed because the lib stays
     * loaded until JVM exit.
     */
    private val arena: Arena = Arena.ofShared()

    /** `true` when the lib has been resolved and `System.load`-ed. */
    fun isLoaded(): Boolean = tryInit()

    /** Symbol lookup for the loaded lib, or `null` if loading failed. */
    fun lookup(): SymbolLookup? {
        tryInit()
        return loadedLookup
    }

    @Synchronized
    private fun tryInit(): Boolean {
        if (initialized) return loadedLookup != null
        initialized = true

        val resourcePath = resolveResourcePath() ?: return false
        val loader = NativeLibraryLoader::class.java.classLoader
            ?: ClassLoader.getSystemClassLoader()

        val stream = loader.getResourceAsStream(resourcePath) ?: return false
        val tmpDir: Path = Files.createTempDirectory("skainet-native-")
        tmpDir.toFile().deleteOnExit()
        val libFile = tmpDir.resolve(resourcePath.substringAfterLast('/'))
        stream.use { Files.copy(it, libFile, StandardCopyOption.REPLACE_EXISTING) }
        libFile.toFile().deleteOnExit()

        return runCatching {
            System.load(libFile.toAbsolutePath().toString())
            loadedLookup = SymbolLookup.libraryLookup(libFile, arena)
            true
        }.getOrElse { false }
    }

    private fun resolveResourcePath(): String? {
        val osTag = osTag() ?: return null
        val archTag = archTag() ?: return null
        val libName = when (osTag) {
            "linux" -> "libskainet_kernels.so"
            "macos" -> "libskainet_kernels.dylib"
            "windows" -> "skainet_kernels.dll"
            else -> return null
        }
        return "native/$osTag-$archTag/$libName"
    }

    private fun osTag(): String? {
        val os = System.getProperty("os.name")?.lowercase() ?: return null
        return when {
            os.contains("linux") -> "linux"
            os.contains("mac") || os.contains("darwin") -> "macos"
            os.contains("windows") -> "windows"
            else -> null
        }
    }

    private fun archTag(): String? {
        val arch = System.getProperty("os.arch")?.lowercase() ?: return null
        return when (arch) {
            "x86_64", "amd64" -> "x86_64"
            "aarch64", "arm64" -> "arm64"
            else -> null
        }
    }
}
