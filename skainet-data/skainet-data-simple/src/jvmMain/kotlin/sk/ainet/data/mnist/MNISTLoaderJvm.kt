package sk.ainet.data.mnist

import sk.ainet.data.common.JvmDatasetSourceReader

/**
 * JVM implementation of the MNIST loader.
 *
 * @property config The configuration for the MNIST loader.
 */
public class MNISTLoaderJvm(config: MNISTLoaderConfig) : MNISTLoaderCommon(config) {
    private val sources = JvmDatasetSourceReader(config.cacheDir, config.useCache)

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
