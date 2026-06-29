package sk.ainet.data.fashionmnist

import sk.ainet.data.common.JvmDatasetSourceReader

/**
 * JVM implementation of the Fashion-MNIST loader.
 *
 * @property config The configuration for the Fashion-MNIST loader.
 */
public class FashionMNISTLoaderJvm(config: FashionMNISTLoaderConfig) : FashionMNISTLoaderCommon(config) {
    private val sources = JvmDatasetSourceReader(
        cacheDir = config.cacheDir,
        useCache = config.useCache,
        huggingFaceTokenProvider = config.huggingFaceTokenProvider,
        useEnvironmentHuggingFaceToken = config.useEnvironmentHuggingFaceToken
    )

    /**
     * Resolves, caches, and decompresses a file when needed.
     *
     * @param url The URL to download from.
     * @param filename The name of the file to save.
     * @return The bytes of the decompressed file.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray {
        return sources.readGzipDecoded(url)
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
