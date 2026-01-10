package sk.ainet.data.mnist

/**
 * Common entry points for obtaining MNIST datasets across platforms.
 *
 * Provides a tiny factory to get the platform-specific MNISTLoader and
 * convenience suspend functions to download and parse the train/test sets.
 */
public object MNIST {
    /**
     * Create a platform-specific MNISTLoader using the provided config.
     */
    public fun loader(config: MNISTLoaderConfig = MNISTLoaderConfig()): MNISTLoader = createMNISTLoader(config)

    /**
     * Download (with caching if supported) and return the MNIST training dataset.
     */
    public suspend fun loadTrain(config: MNISTLoaderConfig = MNISTLoaderConfig()): MNISTDataset =
        loader(config).loadTrainingData()

    /**
     * Download (with caching if supported) and return the MNIST test dataset.
     */
    public suspend fun loadTest(config: MNISTLoaderConfig = MNISTLoaderConfig()): MNISTDataset =
        loader(config).loadTestData()
}

/**
 * Expect/actual factory function implemented per platform that returns the concrete
 * MNISTLoader implementation (e.g., MNISTLoaderJvm, MNISTLoaderAndroid, ...).
 */
public expect fun createMNISTLoader(config: MNISTLoaderConfig): MNISTLoader
