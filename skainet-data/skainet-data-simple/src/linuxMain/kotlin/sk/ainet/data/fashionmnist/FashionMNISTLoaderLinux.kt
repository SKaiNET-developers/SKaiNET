package sk.ainet.data.fashionmnist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

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
            error("FashionMNISTLoaderLinux.downloadAndCacheFile is not implemented yet.")
        }

    public companion object {
        public fun create(): FashionMNISTLoaderLinux = FashionMNISTLoaderLinux(FashionMNISTLoaderConfig())
        public fun create(cacheDir: String): FashionMNISTLoaderLinux = FashionMNISTLoaderLinux(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderLinux = FashionMNISTLoaderLinux(config)
    }
}
