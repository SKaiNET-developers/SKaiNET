package sk.ainet.data.cifar10

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * macOS implementation of the CIFAR-10 loader.
 * Placeholder implementation.
 *
 * @property config The configuration for the CIFAR-10 loader.
 */
public class CIFAR10LoaderMacos(config: CIFAR10LoaderConfig) : CIFAR10LoaderCommon(config) {

    override suspend fun downloadAndExtractBatch(batchFilename: String): ByteArray =
        withContext(Dispatchers.Default) {
            error("CIFAR10LoaderMacos is not implemented yet.")
        }

    public companion object {
        public fun create(): CIFAR10LoaderMacos = CIFAR10LoaderMacos(CIFAR10LoaderConfig())
        public fun create(cacheDir: String): CIFAR10LoaderMacos = CIFAR10LoaderMacos(CIFAR10LoaderConfig(cacheDir = cacheDir))
        public fun create(config: CIFAR10LoaderConfig): CIFAR10LoaderMacos = CIFAR10LoaderMacos(config)
    }
}
