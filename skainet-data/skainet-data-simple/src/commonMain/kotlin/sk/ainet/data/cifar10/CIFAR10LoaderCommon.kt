package sk.ainet.data.cifar10

/**
 * Abstract base class for CIFAR-10 loaders that implements common functionality.
 *
 * CIFAR-10 binary format:
 * - Each image is stored as: 1 byte label + 3072 bytes pixel data
 * - Pixel data is in channel-first format: 1024 red + 1024 green + 1024 blue
 * - Training data is split across 5 batch files (10,000 images each)
 * - Test data is in a single batch file (10,000 images)
 *
 * @property config The configuration for the CIFAR-10 loader.
 */
public abstract class CIFAR10LoaderCommon(public val config: CIFAR10LoaderConfig) : CIFAR10Loader {

    /**
     * Loads the CIFAR-10 training dataset by combining all 5 training batches.
     *
     * @return The CIFAR-10 training dataset (50,000 images).
     */
    override suspend fun loadTrainingData(): CIFAR10Dataset {
        val allImages = mutableListOf<CIFAR10Image>()

        for (batchFilename in CIFAR10Constants.TRAINING_BATCH_FILENAMES) {
            val batchBytes = downloadAndExtractBatch(batchFilename)
            val batchImages = parseBatch(batchBytes)
            allImages.addAll(batchImages)
        }

        return CIFAR10Dataset(allImages)
    }

    /**
     * Loads the CIFAR-10 test dataset.
     *
     * @return The CIFAR-10 test dataset (10,000 images).
     */
    override suspend fun loadTestData(): CIFAR10Dataset {
        val batchBytes = downloadAndExtractBatch(CIFAR10Constants.TEST_BATCH_FILENAME)
        val images = parseBatch(batchBytes)
        return CIFAR10Dataset(images)
    }

    /**
     * Downloads the CIFAR-10 archive and extracts the specified batch file.
     *
     * @param batchFilename The name of the batch file to extract.
     * @return The bytes of the extracted batch file.
     */
    protected abstract suspend fun downloadAndExtractBatch(batchFilename: String): ByteArray

    /**
     * Parses a CIFAR-10 batch file into a list of images.
     *
     * Binary format per image: 1 byte label + 3072 bytes pixel data
     * Pixel data is channel-first: 1024 R + 1024 G + 1024 B
     *
     * @param batchBytes The bytes of the batch file.
     * @return The list of parsed CIFAR-10 images.
     */
    protected fun parseBatch(batchBytes: ByteArray): List<CIFAR10Image> {
        val bytesPerImage = 1 + CIFAR10Constants.IMAGE_BYTES  // 1 label + 3072 pixels
        val numImages = batchBytes.size / bytesPerImage

        if (batchBytes.size % bytesPerImage != 0) {
            throw IllegalArgumentException(
                "Invalid batch file size: ${batchBytes.size} bytes. " +
                "Expected multiple of $bytesPerImage bytes."
            )
        }

        val images = mutableListOf<CIFAR10Image>()

        for (i in 0 until numImages) {
            val offset = i * bytesPerImage

            // First byte is the label
            val label = batchBytes[offset]

            // Next 3072 bytes are pixel data (channel-first: R, G, B)
            val imageData = ByteArray(CIFAR10Constants.IMAGE_BYTES)
            for (j in 0 until CIFAR10Constants.IMAGE_BYTES) {
                imageData[j] = batchBytes[offset + 1 + j]
            }

            images.add(CIFAR10Image(imageData, label))
        }

        return images
    }
}
