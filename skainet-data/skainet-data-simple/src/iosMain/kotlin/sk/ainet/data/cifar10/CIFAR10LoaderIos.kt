package sk.ainet.data.cifar10

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * iOS implementation of the CIFAR-10 loader.
 * Placeholder implementation - tar.gz extraction not yet implemented for iOS.
 *
 * @property config The configuration for the CIFAR-10 loader.
 */
public class CIFAR10LoaderIos(config: CIFAR10LoaderConfig) : CIFAR10LoaderCommon(config) {

    override suspend fun downloadAndExtractBatch(batchFilename: String): ByteArray =
        withContext(Dispatchers.Default) {
            error("CIFAR10LoaderIos is not fully implemented yet. Tar.gz extraction requires native implementation.")
        }

    public companion object {
        public fun create(): CIFAR10LoaderIos = CIFAR10LoaderIos(CIFAR10LoaderConfig())
        public fun create(cacheDir: String): CIFAR10LoaderIos = CIFAR10LoaderIos(CIFAR10LoaderConfig(cacheDir = cacheDir))
        public fun create(config: CIFAR10LoaderConfig): CIFAR10LoaderIos = CIFAR10LoaderIos(config)
    }
}
