package sk.ainet.data.mnist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Linux implementation of the MNIST loader.
 * Minimal placeholder to satisfy expect/actual.
 */
public class MNISTLoaderLinux(config: MNISTLoaderConfig) : MNISTLoaderCommon(config) {

    /**
     * Placeholder: not implemented yet on Linux in this module.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray =
        withContext(Dispatchers.Default) {
            error("MNISTLoaderLinux.downloadAndCacheFile is not implemented yet.")
        }

    public companion object {
        public fun create(): MNISTLoaderLinux = MNISTLoaderLinux(MNISTLoaderConfig())
        public fun create(cacheDir: String): MNISTLoaderLinux = MNISTLoaderLinux(MNISTLoaderConfig(cacheDir = cacheDir))
        public fun create(config: MNISTLoaderConfig): MNISTLoaderLinux = MNISTLoaderLinux(config)
    }
}
