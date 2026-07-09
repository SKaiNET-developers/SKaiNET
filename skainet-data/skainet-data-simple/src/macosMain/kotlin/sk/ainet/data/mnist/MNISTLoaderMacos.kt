package sk.ainet.data.mnist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.common.unsupportedDatasetLoader

/**
 * macOS implementation of the MNIST loader.
 * Minimal placeholder to satisfy expect/actual; avoids direct FS and HTTP until fully implemented.
 */
public class MNISTLoaderMacos(config: MNISTLoaderConfig) : MNISTLoaderCommon(config) {

    /**
     * Placeholder: not implemented yet on macOS in this module. Throws at runtime if invoked.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray =
        withContext(Dispatchers.Default) {
            unsupportedDatasetLoader(
                dataset = "MNIST",
                target = "macos",
                reason = "gzip decompression and cache materialization are not implemented for this native target"
            )
        }

    public companion object {
        public fun create(): MNISTLoaderMacos = MNISTLoaderMacos(MNISTLoaderConfig())
        public fun create(cacheDir: String): MNISTLoaderMacos = MNISTLoaderMacos(MNISTLoaderConfig(cacheDir = cacheDir))
        public fun create(config: MNISTLoaderConfig): MNISTLoaderMacos = MNISTLoaderMacos(config)
    }
}
