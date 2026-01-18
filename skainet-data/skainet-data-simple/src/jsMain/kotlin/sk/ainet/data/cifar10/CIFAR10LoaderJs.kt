package sk.ainet.data.cifar10

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * JS (browser) implementation of the CIFAR-10 loader.
 * Placeholder implementation - tar.gz extraction not yet implemented for JS.
 *
 * @property config The configuration for the CIFAR-10 loader.
 */
public class CIFAR10LoaderJs(config: CIFAR10LoaderConfig) : CIFAR10LoaderCommon(config) {

    override suspend fun downloadAndExtractBatch(batchFilename: String): ByteArray =
        withContext(Dispatchers.Default) {
            error("CIFAR10LoaderJs is not fully implemented yet. Tar.gz extraction requires JS library support.")
        }

    public companion object {
        public fun create(): CIFAR10LoaderJs = CIFAR10LoaderJs(CIFAR10LoaderConfig())
        public fun create(cacheDir: String): CIFAR10LoaderJs = CIFAR10LoaderJs(CIFAR10LoaderConfig(cacheDir = cacheDir))
        public fun create(config: CIFAR10LoaderConfig): CIFAR10LoaderJs = CIFAR10LoaderJs(config)
    }
}
