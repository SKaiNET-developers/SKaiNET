package sk.ainet.data.fashionmnist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.common.unsupportedDatasetLoader

/**
 * Linux implementation of the Fashion-MNIST loader.
 * Minimal placeholder to satisfy expect/actual.
 */
public class FashionMNISTLoaderLinux(config: FashionMNISTLoaderConfig) : FashionMNISTLoaderCommon(config) {

    /**
     * Placeholder: not implemented yet on Linux in this module.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray =
        withContext(Dispatchers.Default) {
            unsupportedDatasetLoader(
                dataset = "Fashion-MNIST",
                target = "linux",
                reason = "gzip decompression and cache materialization are not implemented for this native target"
            )
        }

    public companion object {
        public fun create(): FashionMNISTLoaderLinux = FashionMNISTLoaderLinux(FashionMNISTLoaderConfig())
        public fun create(cacheDir: String): FashionMNISTLoaderLinux = FashionMNISTLoaderLinux(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderLinux = FashionMNISTLoaderLinux(config)
    }
}
