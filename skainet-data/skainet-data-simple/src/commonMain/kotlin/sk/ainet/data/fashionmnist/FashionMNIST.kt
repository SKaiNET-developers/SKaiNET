package sk.ainet.data.fashionmnist

/**
 * Common entry points for obtaining Fashion-MNIST datasets across platforms.
 *
 * Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of
 * 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
 * associated with a label from 10 classes.
 *
 * It serves as a direct drop-in replacement for the original MNIST dataset for benchmarking
 * machine learning algorithms, as it shares the same image size and structure of training
 * and testing splits.
 *
 * ## Usage
 *
 * ```kotlin
 * // Load training data
 * val trainDataset = FashionMNIST.loadTrain()
 *
 * // Load test data
 * val testDataset = FashionMNIST.loadTest()
 *
 * // Iterate over batches
 * val batchIterator = trainDataset.batchIterator<Int8, Byte>(32)
 * for (batch in batchIterator) {
 *     val images = batch.x[0]  // Shape: [32, 1, 28, 28]
 *     val labels = batch.y     // Shape: [32]
 * }
 * ```
 */
public object FashionMNIST {
    /**
     * Create a platform-specific FashionMNISTLoader using the provided config.
     */
    public fun loader(config: FashionMNISTLoaderConfig = FashionMNISTLoaderConfig()): FashionMNISTLoader =
        createFashionMNISTLoader(config)

    /**
     * Download (with caching if supported) and return the Fashion-MNIST training dataset.
     *
     * @param config Configuration for caching and download behavior.
     * @return The Fashion-MNIST training dataset (60,000 images).
     */
    public suspend fun loadTrain(config: FashionMNISTLoaderConfig = FashionMNISTLoaderConfig()): FashionMNISTDataset =
        loader(config).loadTrainingData()

    /**
     * Download (with caching if supported) and return the Fashion-MNIST test dataset.
     *
     * @param config Configuration for caching and download behavior.
     * @return The Fashion-MNIST test dataset (10,000 images).
     */
    public suspend fun loadTest(config: FashionMNISTLoaderConfig = FashionMNISTLoaderConfig()): FashionMNISTDataset =
        loader(config).loadTestData()
}

/**
 * Expect/actual factory function implemented per platform that returns the concrete
 * FashionMNISTLoader implementation.
 */
public expect fun createFashionMNISTLoader(config: FashionMNISTLoaderConfig): FashionMNISTLoader
