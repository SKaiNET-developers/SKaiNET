package sk.ainet.data.fashionmnist

/**
 * Abstract base class for Fashion-MNIST loaders that implements common functionality.
 *
 * Fashion-MNIST uses the same IDX file format as MNIST, so the parsing logic is identical.
 *
 * @property config The configuration for the Fashion-MNIST loader.
 */
public abstract class FashionMNISTLoaderCommon(public val config: FashionMNISTLoaderConfig) : FashionMNISTLoader {

    /**
     * Loads the Fashion-MNIST training dataset.
     *
     * @return The Fashion-MNIST training dataset.
     */
    override suspend fun loadTrainingData(): FashionMNISTDataset {
        val imagesBytes = downloadAndCacheFile(
            config.trainImagesUri,
            FashionMNISTConstants.TRAIN_IMAGES_FILENAME
        )
        val labelsBytes = downloadAndCacheFile(
            config.trainLabelsUri,
            FashionMNISTConstants.TRAIN_LABELS_FILENAME
        )

        return parseDataset(imagesBytes, labelsBytes)
    }

    /**
     * Loads the Fashion-MNIST test dataset.
     *
     * @return The Fashion-MNIST test dataset.
     */
    override suspend fun loadTestData(): FashionMNISTDataset {
        val imagesBytes = downloadAndCacheFile(
            config.testImagesUri,
            FashionMNISTConstants.TEST_IMAGES_FILENAME
        )
        val labelsBytes = downloadAndCacheFile(
            config.testLabelsUri,
            FashionMNISTConstants.TEST_LABELS_FILENAME
        )

        return parseDataset(imagesBytes, labelsBytes)
    }

    /**
     * Downloads and caches a file.
     *
     * @param url The URL to download from.
     * @param filename The name of the file to save.
     * @return The bytes of the decompressed file.
     */
    protected abstract suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray

    /**
     * Parses the Fashion-MNIST dataset from the images and labels files.
     * Uses the same IDX format as MNIST.
     *
     * @param imagesBytes The bytes of the images file.
     * @param labelsBytes The bytes of the labels file.
     * @return The parsed Fashion-MNIST dataset.
     */
    protected fun parseDataset(imagesBytes: ByteArray, labelsBytes: ByteArray): FashionMNISTDataset {
        // Parse images header
        val imagesMagic = readInt32(imagesBytes, 0)
        if (imagesMagic != 2051) {
            throw IllegalArgumentException("Invalid magic number for images file: $imagesMagic (expected 2051)")
        }

        val numImages = readInt32(imagesBytes, 4)
        val numRows = readInt32(imagesBytes, 8)
        val numCols = readInt32(imagesBytes, 12)

        if (numRows != FashionMNISTConstants.IMAGE_SIZE || numCols != FashionMNISTConstants.IMAGE_SIZE) {
            throw IllegalArgumentException("Invalid image dimensions: $numRows x $numCols (expected 28x28)")
        }

        // Parse labels header
        val labelsMagic = readInt32(labelsBytes, 0)
        if (labelsMagic != 2049) {
            throw IllegalArgumentException("Invalid magic number for labels file: $labelsMagic (expected 2049)")
        }

        val numLabels = readInt32(labelsBytes, 4)

        if (numImages != numLabels) {
            throw IllegalArgumentException("Number of images ($numImages) does not match number of labels ($numLabels)")
        }

        // Create dataset
        val images = mutableListOf<FashionMNISTImage>()

        for (i in 0 until numImages) {
            val imageOffset = 16 + i * FashionMNISTConstants.IMAGE_PIXELS
            val labelOffset = 8 + i

            val image = ByteArray(FashionMNISTConstants.IMAGE_PIXELS)
            for (j in 0 until FashionMNISTConstants.IMAGE_PIXELS) {
                image[j] = imagesBytes[imageOffset + j]
            }

            val label = labelsBytes[labelOffset]

            images.add(FashionMNISTImage(image, label))
        }

        return FashionMNISTDataset(images)
    }

    /**
     * Reads a 32-bit integer from a byte array in big-endian format.
     *
     * @param bytes The byte array.
     * @param offset The offset to read from.
     * @return The 32-bit integer.
     */
    private fun readInt32(bytes: ByteArray, offset: Int): Int {
        return ((bytes[offset].toInt() and 0xFF) shl 24) or
                ((bytes[offset + 1].toInt() and 0xFF) shl 16) or
                ((bytes[offset + 2].toInt() and 0xFF) shl 8) or
                (bytes[offset + 3].toInt() and 0xFF)
    }
}
