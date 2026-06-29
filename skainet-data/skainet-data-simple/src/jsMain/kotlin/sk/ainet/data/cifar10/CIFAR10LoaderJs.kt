package sk.ainet.data.cifar10

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.common.unsupportedDatasetLoader

/**
 * JS (browser) implementation of the CIFAR-10 loader.
 * Placeholder implementation - tar.gz extraction not yet implemented for JS.
 *
 * @property config The configuration for the CIFAR-10 loader.
 */
public class CIFAR10LoaderJs(config: CIFAR10LoaderConfig) : CIFAR10LoaderCommon(config) {

    override suspend fun downloadAndExtractBatch(batchFilename: String): ByteArray =
        withContext(Dispatchers.Default) {
            unsupportedDatasetLoader(
                dataset = "CIFAR-10",
                target = "js",
                reason = "tar.gz extraction is not implemented for this browser target"
            )
        }

    public companion object {
        public fun create(): CIFAR10LoaderJs = CIFAR10LoaderJs(CIFAR10LoaderConfig())
        public fun create(cacheDir: String): CIFAR10LoaderJs = CIFAR10LoaderJs(CIFAR10LoaderConfig(cacheDir = cacheDir))
        public fun create(config: CIFAR10LoaderConfig): CIFAR10LoaderJs = CIFAR10LoaderJs(config)
    }
}
