package sk.ainet.data.fashionmnist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

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
            error("FashionMNISTLoaderMacos.downloadAndCacheFile is not implemented yet. Provide an appleMain implementation or avoid macOS usage for now.")
        }

    public companion object {
        public fun create(): FashionMNISTLoaderMacos = FashionMNISTLoaderMacos(FashionMNISTLoaderConfig())
        public fun create(cacheDir: String): FashionMNISTLoaderMacos = FashionMNISTLoaderMacos(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderMacos = FashionMNISTLoaderMacos(config)
    }
}
