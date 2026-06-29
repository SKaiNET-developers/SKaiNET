package sk.ainet.data.cifar10

import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.data.DataBatch
import sk.ainet.data.Dataset
import sk.ainet.data.common.DatasetHuggingFaceTokenProvider
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Int8
import kotlin.math.min
import kotlin.random.Random

/**
 * Represents a single CIFAR-10 image with its label.
 *
 * CIFAR-10 consists of 60,000 32x32 color images in 10 classes:
 * - 50,000 training images
 * - 10,000 test images
 *
 * Classes:
 * - 0: airplane
 * - 1: automobile
 * - 2: bird
 * - 3: cat
 * - 4: deer
 * - 5: dog
 * - 6: frog
 * - 7: horse
 * - 8: ship
 * - 9: truck
 *
 * @property image The pixel data of the image as a ByteArray (3x32x32 = 3072 bytes, channel-first RGB).
 * @property label The label of the image (0-9).
 */
@Serializable
public data class CIFAR10Image(
    val image: ByteArray,
    val label: Byte
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as CIFAR10Image

        if (!image.contentEquals(other.image)) return false
        if (label != other.label) return false

        return true
    }

    override fun hashCode(): Int {
        var result = image.contentHashCode()
        result = 31 * result + label.toInt()
        return result
    }

    /**
     * Returns the class name for this image's label.
     */
    public fun className(): String = CIFAR10Constants.CLASS_NAMES[label.toInt()]
}

/**
 * CIFAR-10 dataset implementation using Dataset/DataBatch API.
 * - Provides batching as tensors [Int8] with shapes:
 *   x: [batch, 3, 32, 32] (channel-first RGB)
 *   y: [batch] (labels as bytes)
 */
@Serializable
public data class CIFAR10Dataset(
    val images: List<CIFAR10Image>,
    @Transient private val executionContext: ExecutionContext = DefaultDataExecutionContext()
) : Dataset<CIFAR10Image, Float>() {

    override val xSize: Int get() = images.size

    override fun getX(idx: Int): CIFAR10Image = images[idx]

    override fun getY(idx: Int): Float = images[idx].label.toInt().toFloat()

    override fun shuffle(): Dataset<CIFAR10Image, Float> {
        val shuffled = images.toMutableList()
        shuffled.shuffle(Random.Default)
        return CIFAR10Dataset(shuffled, executionContext)
    }

    override fun split(splitRatio: Double): Pair<Dataset<CIFAR10Image, Float>, Dataset<CIFAR10Image, Float>> {
        require(splitRatio > 0.0 && splitRatio < 1.0) { "splitRatio must be in (0,1)" }
        val splitIndex = (images.size * splitRatio).toInt()
        val first = images.subList(0, splitIndex)
        val second = images.subList(splitIndex, images.size)
        return CIFAR10Dataset(first, executionContext) to CIFAR10Dataset(second, executionContext)
    }

    /**
     * Creates a DataBatch with memory-efficient Int8 tensors from raw byte arrays.
     */
    override fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V> {
        val actualLen = min(batchLength, xSize - batchStart)
        val batchImages = images.subList(batchStart, batchStart + actualLen)

        // Concatenate raw image bytes (no normalization) for memory efficiency
        val xData = ByteArray(actualLen * CIFAR10Constants.IMAGE_BYTES)
        var offset = 0
        for (sample in batchImages) {
            val bytes = sample.image
            bytes.copyInto(xData, destinationOffset = offset, startIndex = 0, endIndex = CIFAR10Constants.IMAGE_BYTES)
            offset += CIFAR10Constants.IMAGE_BYTES
        }

        // Shape as [batch, 3, 32, 32] (channel-first)
        val xShape = Shape(actualLen, CIFAR10Constants.NUM_CHANNELS, CIFAR10Constants.IMAGE_SIZE, CIFAR10Constants.IMAGE_SIZE)
        val xTensor: Tensor<Int8, Byte> = executionContext.fromByteArray(xShape, Int8::class, xData)

        // Labels as bytes (memory-efficient)
        val yData = ByteArray(actualLen) { idx -> batchImages[idx].label }
        val yShape = Shape(actualLen)
        val yTensor: Tensor<Int8, Byte> = executionContext.fromByteArray(yShape, Int8::class, yData)

        // DataBatch expects array of input tensors; we provide single input
        val xArray: Array<Tensor<Int8, Byte>> = arrayOf(xTensor)

        @Suppress("UNCHECKED_CAST")
        return DataBatch(xArray as Array<Tensor<T, V>>, yTensor as Tensor<T, V>)
    }

    /**
     * Returns a subset of the dataset.
     */
    public fun subset(fromIndex: Int, toIndex: Int): CIFAR10Dataset {
        return CIFAR10Dataset(images.subList(fromIndex, toIndex), executionContext)
    }
}

/**
 * Configuration for the CIFAR-10 loader.
 *
 * @property cacheDir The directory where downloaded files will be cached.
 * @property useCache Whether to use cached files if available.
 */
public data class CIFAR10LoaderConfig(
    val cacheDir: String = "cifar10-data",
    val useCache: Boolean = true,
    val archiveUri: String = CIFAR10Constants.DOWNLOAD_URL,
    val huggingFaceTokenProvider: DatasetHuggingFaceTokenProvider? = null,
    val useEnvironmentHuggingFaceToken: Boolean = false
)

/**
 * Constants for the CIFAR-10 dataset.
 */
public object CIFAR10Constants {
    public const val IMAGE_SIZE: Int = 32
    public const val NUM_CHANNELS: Int = 3
    public const val IMAGE_BYTES: Int = NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE  // 3072

    public const val NUM_TRAINING_IMAGES: Int = 50000
    public const val NUM_TEST_IMAGES: Int = 10000
    public const val IMAGES_PER_BATCH: Int = 10000

    // CIFAR-10 binary format filenames
    public const val DATA_BATCH_1_FILENAME: String = "data_batch_1.bin"
    public const val DATA_BATCH_2_FILENAME: String = "data_batch_2.bin"
    public const val DATA_BATCH_3_FILENAME: String = "data_batch_3.bin"
    public const val DATA_BATCH_4_FILENAME: String = "data_batch_4.bin"
    public const val DATA_BATCH_5_FILENAME: String = "data_batch_5.bin"
    public const val TEST_BATCH_FILENAME: String = "test_batch.bin"

    // Archive filename
    public const val ARCHIVE_FILENAME: String = "cifar-10-binary.tar.gz"

    // CIFAR-10 download URL (University of Toronto)
    public const val DOWNLOAD_URL: String =
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

    /**
     * Class names for CIFAR-10 labels.
     */
    public val CLASS_NAMES: List<String> = listOf(
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    )

    /**
     * Training batch filenames in order.
     */
    public val TRAINING_BATCH_FILENAMES: List<String> = listOf(
        DATA_BATCH_1_FILENAME,
        DATA_BATCH_2_FILENAME,
        DATA_BATCH_3_FILENAME,
        DATA_BATCH_4_FILENAME,
        DATA_BATCH_5_FILENAME
    )
}

/**
 * Interface for the CIFAR-10 loader.
 */
public interface CIFAR10Loader {
    /**
     * Loads the CIFAR-10 training dataset.
     *
     * @return The CIFAR-10 training dataset (50,000 images).
     */
    public suspend fun loadTrainingData(): CIFAR10Dataset

    /**
     * Loads the CIFAR-10 test dataset.
     *
     * @return The CIFAR-10 test dataset (10,000 images).
     */
    public suspend fun loadTestData(): CIFAR10Dataset
}
