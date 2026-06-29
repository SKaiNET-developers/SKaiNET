package sk.ainet.data.mnist

/**
 * Abstract base class for MNIST loaders that implements common functionality.
 *
 * @property config The configuration for the MNIST loader.
 */
public abstract class MNISTLoaderCommon(public val config: MNISTLoaderConfig) : MNISTLoader {

    /**
     * Loads the MNIST training dataset.
     *
     * @return The MNIST training dataset.
     */
    override suspend fun loadTrainingData(): MNISTDataset {
        val imagesBytes = downloadAndCacheFile(
            config.trainImagesUri,
            MNISTConstants.TRAIN_IMAGES_FILENAME
        )
        val labelsBytes = downloadAndCacheFile(
            config.trainLabelsUri,
            MNISTConstants.TRAIN_LABELS_FILENAME
        )

        return parseDataset(imagesBytes, labelsBytes)
    }

    /**
     * Loads the MNIST test dataset.
     *
     * @return The MNIST test dataset.
     */
    override suspend fun loadTestData(): MNISTDataset {
        val imagesBytes = downloadAndCacheFile(
            config.testImagesUri,
            MNISTConstants.TEST_IMAGES_FILENAME
        )
        val labelsBytes = downloadAndCacheFile(
            config.testLabelsUri,
            MNISTConstants.TEST_LABELS_FILENAME
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
     * Parses the MNIST dataset from the images and labels files.
     *
     * @param imagesBytes The bytes of the images file.
     * @param labelsBytes The bytes of the labels file.
     * @return The parsed MNIST dataset.
     */
    protected fun parseDataset(imagesBytes: ByteArray, labelsBytes: ByteArray): MNISTDataset {
        // Parse images
        val imagesMagic = readInt32(imagesBytes, 0)
        if (imagesMagic != 2051) {
            throw IllegalArgumentException("Invalid magic number for images file: $imagesMagic")
        }

        val numImages = readInt32(imagesBytes, 4)
        val numRows = readInt32(imagesBytes, 8)
        val numCols = readInt32(imagesBytes, 12)

        if (numRows != MNISTConstants.IMAGE_SIZE || numCols != MNISTConstants.IMAGE_SIZE) {
            throw IllegalArgumentException("Invalid image dimensions: $numRows x $numCols")
        }

        // Parse labels
        val labelsMagic = readInt32(labelsBytes, 0)
        if (labelsMagic != 2049) {
            throw IllegalArgumentException("Invalid magic number for labels file: $labelsMagic")
        }

        val numLabels = readInt32(labelsBytes, 4)

        if (numImages != numLabels) {
            throw IllegalArgumentException("Number of images ($numImages) does not match number of labels ($numLabels)")
        }

        // Create dataset
        val images = mutableListOf<MNISTImage>()

        for (i in 0 until numImages) {
            val imageOffset = 16 + i * MNISTConstants.IMAGE_PIXELS
            val labelOffset = 8 + i

            val image = ByteArray(MNISTConstants.IMAGE_PIXELS)
            for (j in 0 until MNISTConstants.IMAGE_PIXELS) {
                image[j] = imagesBytes[imageOffset + j]
            }

            val label = labelsBytes[labelOffset]

            images.add(MNISTImage(image, label))
        }

        return MNISTDataset(images)
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
