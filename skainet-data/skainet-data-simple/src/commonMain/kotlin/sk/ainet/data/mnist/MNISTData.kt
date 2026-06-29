package sk.ainet.data.mnist

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
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.math.min
import kotlin.random.Random

/**
 * Represents a single MNIST image with its label.
 *
 * @property image The pixel data of the image as a ByteArray (28x28 pixels).
 * @property label The label of the image (0-9).
 */
@Serializable
public data class MNISTImage(
    val image: ByteArray,
    val label: Byte
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as MNISTImage

        if (!image.contentEquals(other.image)) return false
        if (label != other.label) return false

        return true
    }

    override fun hashCode(): Int {
        var result = image.contentHashCode()
        result = 31 * result + label.toInt()
        return result
    }
}

/**
 * MNIST dataset implementation using Dataset/DataBatch API.
 * - Provides batching as tensors [FP32] with shapes:
 *   x: [batch, 1, 28, 28] (normalized 0..1)
 *   y: [batch] (labels as floats)
 */
@Serializable
public data class MNISTDataset(
    val images: List<MNISTImage>,
    @Transient private val executionContext: ExecutionContext = DefaultDataExecutionContext()
) : Dataset<MNISTImage, Float>() {

    override val xSize: Int get() = images.size

    override fun getX(idx: Int): MNISTImage = images[idx]

    override fun getY(idx: Int): Float = images[idx].label.toInt().toFloat()

    override fun shuffle(): Dataset<MNISTImage, Float> {
        val shuffled = images.toMutableList()
        shuffled.shuffle(Random.Default)
        return MNISTDataset(shuffled, executionContext)
    }

    override fun split(splitRatio: Double): Pair<Dataset<MNISTImage, Float>, Dataset<MNISTImage, Float>> {
        require(splitRatio > 0.0 && splitRatio < 1.0) { "splitRatio must be in (0,1)" }
        val splitIndex = (images.size * splitRatio).toInt()
        val first = images.subList(0, splitIndex)
        val second = images.subList(splitIndex, images.size)
        return MNISTDataset(first, executionContext) to MNISTDataset(second, executionContext)
    }

    /**
     * Creates a DataBatch with memory-efficient Int8 tensors from raw byte arrays.
     */
    override fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V> {
        val actualLen = min(batchLength, xSize - batchStart)
        val batchIndices = IntArray(actualLen) { offset -> batchStart + offset }
        return createDataBatchForIndices(batchIndices)
    }

    /**
     * Creates a DataBatch from arbitrary sample indices. This keeps shuffled and
     * filtered dataset views compatible with tensor batching.
     */
    override fun <T : DType, V> createIndexedDataBatch(indices: IntArray): DataBatch<T, V> =
        createDataBatchForIndices(indices)

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> createDataBatchForIndices(indices: IntArray): DataBatch<T, V> {
        val batchImages = indices.map { images[it] }

        // Concatenate raw image bytes (no normalization) for memory efficiency
        val xData = ByteArray(indices.size * MNISTConstants.IMAGE_PIXELS)
        var offset = 0
        for (sample in batchImages) {
            val bytes = sample.image
            bytes.copyInto(xData, destinationOffset = offset, startIndex = 0, endIndex = MNISTConstants.IMAGE_PIXELS)
            offset += MNISTConstants.IMAGE_PIXELS
        }

        // Shape as [batch, 1, 28, 28]
        val xShape = Shape(indices.size, 1, MNISTConstants.IMAGE_SIZE, MNISTConstants.IMAGE_SIZE)
        val xTensor: Tensor<Int8, Byte> = executionContext.fromByteArray(xShape, Int8::class, xData)

        // Labels as bytes (memory-efficient). Keep as Int8 to satisfy DataBatch single dtype requirement
        val yData = ByteArray(indices.size) { idx -> batchImages[idx].label }
        val yShape = Shape(indices.size)
        val yTensor: Tensor<Int8, Byte> = executionContext.fromByteArray(yShape, Int8::class, yData)

        // DataBatch expects array of input tensors; we provide single input
        val xArray: Array<Tensor<Int8, Byte>> = arrayOf(xTensor)

        return DataBatch(xArray as Array<Tensor<T, V>>, yTensor as Tensor<T, V>, indices = indices.copyOf())
    }

    /**
     * Returns a subset of the dataset.
     */
    public fun subset(fromIndex: Int, toIndex: Int): MNISTDataset {
        return MNISTDataset(images.subList(fromIndex, toIndex), executionContext)
    }
}

/**
 * Configuration for the MNIST loader.
 *
 * @property cacheDir The directory where downloaded files will be cached.
 * @property useCache Whether to use cached files if available.
 */
public data class MNISTLoaderConfig(
    val cacheDir: String = "mnist-data",
    val useCache: Boolean = true,
    val trainImagesUri: String = MNISTConstants.TRAIN_IMAGES_URL,
    val trainLabelsUri: String = MNISTConstants.TRAIN_LABELS_URL,
    val testImagesUri: String = MNISTConstants.TEST_IMAGES_URL,
    val testLabelsUri: String = MNISTConstants.TEST_LABELS_URL,
    val huggingFaceTokenProvider: DatasetHuggingFaceTokenProvider? = null,
    val useEnvironmentHuggingFaceToken: Boolean = false
)

/**
 * Constants for the MNIST dataset.
 */
public object MNISTConstants {
    public const val IMAGE_SIZE: Int = 28
    public const val IMAGE_PIXELS: Int = IMAGE_SIZE * IMAGE_SIZE

    public const val TRAIN_IMAGES_FILENAME: String = "train-images-idx3-ubyte.gz"
    public const val TRAIN_LABELS_FILENAME: String = "train-labels-idx1-ubyte.gz"
    public const val TEST_IMAGES_FILENAME: String = "t10k-images-idx3-ubyte.gz"
    public const val TEST_LABELS_FILENAME: String = "t10k-labels-idx1-ubyte.gz"

    public const val TRAIN_IMAGES_URL: String =
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
    public const val TRAIN_LABELS_URL: String =
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
    public const val TEST_IMAGES_URL: String = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
    public const val TEST_LABELS_URL: String = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
}

/**
 * Interface for the MNIST loader.
 */
public interface MNISTLoader {
    /**
     * Loads the MNIST training dataset.
     *
     * @return The MNIST training dataset.
     */
    public suspend fun loadTrainingData(): MNISTDataset

    /**
     * Loads the MNIST test dataset.
     *
     * @return The MNIST test dataset.
     */
    public suspend fun loadTestData(): MNISTDataset
}
