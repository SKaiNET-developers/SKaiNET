package sk.ainet.data.cifar10

/**
 * Common entry points for obtaining CIFAR-10 datasets across platforms.
 *
 * CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes:
 * airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
 * There are 50,000 training images and 10,000 test images.
 *
 * The images are stored in channel-first format (3x32x32 = 3072 bytes per image).
 *
 * ## Usage
 *
 * ```kotlin
 * // Load training data
 * val trainDataset = CIFAR10.loadTrain()
 *
 * // Load test data
 * val testDataset = CIFAR10.loadTest()
 *
 * // Iterate over batches
 * val batchIterator = trainDataset.batchIterator<Int8, Byte>(32)
 * for (batch in batchIterator) {
 *     val images = batch.x[0]  // Shape: [32, 3, 32, 32]
 *     val labels = batch.y     // Shape: [32]
 * }
 *
 * // Get class name for a sample
 * val sample = trainDataset.getX(0)
 * println("Class: ${sample.className()}")  // e.g., "airplane"
 * ```
 */
public object CIFAR10 {
    /**
     * Create a platform-specific CIFAR10Loader using the provided config.
     */
    public fun loader(config: CIFAR10LoaderConfig = CIFAR10LoaderConfig()): CIFAR10Loader =
        createCIFAR10Loader(config)

    /**
     * Download (with caching if supported) and return the CIFAR-10 training dataset.
     *
     * @param config Configuration for caching and download behavior.
     * @return The CIFAR-10 training dataset (50,000 images).
     */
    public suspend fun loadTrain(config: CIFAR10LoaderConfig = CIFAR10LoaderConfig()): CIFAR10Dataset =
        loader(config).loadTrainingData()

    /**
     * Download (with caching if supported) and return the CIFAR-10 test dataset.
     *
     * @param config Configuration for caching and download behavior.
     * @return The CIFAR-10 test dataset (10,000 images).
     */
    public suspend fun loadTest(config: CIFAR10LoaderConfig = CIFAR10LoaderConfig()): CIFAR10Dataset =
        loader(config).loadTestData()
}

/**
 * Expect/actual factory function implemented per platform that returns the concrete
 * CIFAR10Loader implementation.
 */
public expect fun createCIFAR10Loader(config: CIFAR10LoaderConfig): CIFAR10Loader
