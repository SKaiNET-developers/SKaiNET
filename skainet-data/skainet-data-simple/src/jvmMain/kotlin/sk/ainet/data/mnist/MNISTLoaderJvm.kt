package sk.ainet.data.mnist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.source.CachePolicy
import sk.ainet.data.source.DataSourceRequest
import sk.ainet.data.source.JvmDataSourceResolver
import java.io.ByteArrayInputStream
import java.util.zip.GZIPInputStream
import java.io.File

/**
 * JVM implementation of the MNIST loader.
 *
 * @property config The configuration for the MNIST loader.
 */
public class MNISTLoaderJvm(config: MNISTLoaderConfig) : MNISTLoaderCommon(config) {
    private val resolver = JvmDataSourceResolver(File(config.cacheDir, "sources"))

    /**
     * Resolves, caches, and decompresses a file when needed.
     *
     * @param url The URL to download from.
     * @param filename The name of the file to save.
     * @return The bytes of the decompressed file.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray = withContext(Dispatchers.IO) {
        val artifact = resolver.resolve(
            DataSourceRequest(
                uri = url,
                cachePolicy = if (config.useCache) CachePolicy.Use else CachePolicy.Refresh
            )
        )
        return@withContext maybeGunzip(artifact.readBytes())
    }

    private fun maybeGunzip(bytes: ByteArray): ByteArray {
        if (!bytes.isGzip()) return bytes
        return GZIPInputStream(ByteArrayInputStream(bytes)).use { it.readBytes() }
    }

    private fun ByteArray.isGzip(): Boolean {
        return size >= 2 && this[0] == 0x1f.toByte() && this[1] == 0x8b.toByte()
    }

    public companion object {
        /**
         * Creates a new instance of MNISTLoaderJvm with the default configuration.
         *
         * @return A new instance of MNISTLoaderJvm.
         */
        public fun create(): MNISTLoaderJvm {
            return MNISTLoaderJvm(MNISTLoaderConfig())
        }

        /**
         * Creates a new instance of MNISTLoaderJvm with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of MNISTLoaderJvm.
         */
        public fun create(cacheDir: String): MNISTLoaderJvm {
            return MNISTLoaderJvm(MNISTLoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of MNISTLoaderJvm with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of MNISTLoaderJvm.
         */
        public fun create(config: MNISTLoaderConfig): MNISTLoaderJvm {
            return MNISTLoaderJvm(config)
        }
    }

}
