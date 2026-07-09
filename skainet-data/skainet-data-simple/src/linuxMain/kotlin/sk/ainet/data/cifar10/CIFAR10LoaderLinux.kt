package sk.ainet.data.cifar10

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.common.unsupportedDatasetLoader

/**
 * Linux implementation of the CIFAR-10 loader.
 * Placeholder implementation.
 *
 * @property config The configuration for the CIFAR-10 loader.
 */
public class CIFAR10LoaderLinux(config: CIFAR10LoaderConfig) : CIFAR10LoaderCommon(config) {

    override suspend fun downloadAndExtractBatch(batchFilename: String): ByteArray =
        withContext(Dispatchers.Default) {
            unsupportedDatasetLoader(
                dataset = "CIFAR-10",
                target = "linux",
                reason = "tar.gz extraction is not implemented for this native target"
            )
        }

    public companion object {
        public fun create(): CIFAR10LoaderLinux = CIFAR10LoaderLinux(CIFAR10LoaderConfig())
        public fun create(cacheDir: String): CIFAR10LoaderLinux = CIFAR10LoaderLinux(CIFAR10LoaderConfig(cacheDir = cacheDir))
        public fun create(config: CIFAR10LoaderConfig): CIFAR10LoaderLinux = CIFAR10LoaderLinux(config)
    }
}
