package sk.ainet.data.fashionmnist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.source.CachePolicy
import sk.ainet.data.source.DataSourceRequest
import sk.ainet.data.source.JvmDataSourceResolver
import java.io.ByteArrayInputStream
import java.util.zip.GZIPInputStream
import java.io.File

/**
 * JVM implementation of the Fashion-MNIST loader.
 *
 * @property config The configuration for the Fashion-MNIST loader.
 */
public class FashionMNISTLoaderJvm(config: FashionMNISTLoaderConfig) : FashionMNISTLoaderCommon(config) {
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
         * Creates a new instance of FashionMNISTLoaderJvm with the default configuration.
         *
         * @return A new instance of FashionMNISTLoaderJvm.
         */
        public fun create(): FashionMNISTLoaderJvm {
            return FashionMNISTLoaderJvm(FashionMNISTLoaderConfig())
        }

        /**
         * Creates a new instance of FashionMNISTLoaderJvm with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of FashionMNISTLoaderJvm.
         */
        public fun create(cacheDir: String): FashionMNISTLoaderJvm {
            return FashionMNISTLoaderJvm(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of FashionMNISTLoaderJvm with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of FashionMNISTLoaderJvm.
         */
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderJvm {
            return FashionMNISTLoaderJvm(config)
        }
    }

}
