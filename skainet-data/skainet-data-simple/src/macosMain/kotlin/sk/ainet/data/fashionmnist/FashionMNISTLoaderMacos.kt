package sk.ainet.data.fashionmnist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.common.unsupportedDatasetLoader

/**
 * macOS implementation of the Fashion-MNIST loader.
 * Minimal placeholder to satisfy expect/actual; avoids direct FS and HTTP until fully implemented.
 */
public class FashionMNISTLoaderMacos(config: FashionMNISTLoaderConfig) : FashionMNISTLoaderCommon(config) {

    /**
     * Placeholder: not implemented yet on macOS in this module. Throws at runtime if invoked.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray =
        withContext(Dispatchers.Default) {
            unsupportedDatasetLoader(
                dataset = "Fashion-MNIST",
                target = "macos",
                reason = "gzip decompression and cache materialization are not implemented for this native target"
            )
        }

    public companion object {
        public fun create(): FashionMNISTLoaderMacos = FashionMNISTLoaderMacos(FashionMNISTLoaderConfig())
        public fun create(cacheDir: String): FashionMNISTLoaderMacos = FashionMNISTLoaderMacos(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderMacos = FashionMNISTLoaderMacos(config)
    }
}
