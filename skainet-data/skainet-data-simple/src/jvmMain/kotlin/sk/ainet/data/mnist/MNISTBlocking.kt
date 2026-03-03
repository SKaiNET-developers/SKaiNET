@file:JvmName("MNISTBlocking")

package sk.ainet.data.mnist

import kotlinx.coroutines.runBlocking

/**
 * Blocking (non-suspend) MNIST data loading for Java interop.
 *
 * These methods wrap the suspend functions in [MNIST] using `runBlocking`,
 * making them callable from plain Java code.
 *
 * Example usage from Java:
 * ```java
 * MNISTDataset train = MNISTBlocking.loadTrain();
 * MNISTDataset test = MNISTBlocking.loadTest();
 * ```
 */
public object MNISTBlocking {

    /**
     * Download and return the MNIST training dataset (blocking).
     *
     * @param config Loader configuration. Defaults to standard settings.
     * @return The MNIST training dataset.
     */
    @JvmStatic
    @JvmOverloads
    public fun loadTrain(config: MNISTLoaderConfig = MNISTLoaderConfig()): MNISTDataset = runBlocking {
        MNIST.loadTrain(config)
    }

    /**
     * Download and return the MNIST test dataset (blocking).
     *
     * @param config Loader configuration. Defaults to standard settings.
     * @return The MNIST test dataset.
     */
    @JvmStatic
    @JvmOverloads
    public fun loadTest(config: MNISTLoaderConfig = MNISTLoaderConfig()): MNISTDataset = runBlocking {
        MNIST.loadTest(config)
    }
}
