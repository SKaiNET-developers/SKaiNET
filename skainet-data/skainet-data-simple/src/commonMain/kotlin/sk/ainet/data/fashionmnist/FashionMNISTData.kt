package sk.ainet.data.fashionmnist

import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.data.DataBatch
import sk.ainet.data.Dataset
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Int8
import kotlin.math.min
import kotlin.random.Random

/**
 * Represents a single Fashion-MNIST image with its label.
 *
 * Fashion-MNIST is a dataset of Zalando's article images consisting of:
 * - 60,000 training examples
 * - 10,000 test examples
 * - Each example is a 28x28 grayscale image
 * - Associated with a label from 10 classes
 *
 * Classes:
 * - 0: T-shirt/top
 * - 1: Trouser
 * - 2: Pullover
 * - 3: Dress
 * - 4: Coat
 * - 5: Sandal
 * - 6: Shirt
 * - 7: Sneaker
 * - 8: Bag
 * - 9: Ankle boot
 *
 * @property image The pixel data of the image as a ByteArray (28x28 pixels).
 * @property label The label of the image (0-9).
 */
@Serializable
public data class FashionMNISTImage(
    val image: ByteArray,
    val label: Byte
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as FashionMNISTImage

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
    public fun className(): String = FashionMNISTConstants.CLASS_NAMES[label.toInt()]
}

/**
 * Fashion-MNIST dataset implementation using Dataset/DataBatch API.
 * - Provides batching as tensors [Int8] with shapes:
 *   x: [batch, 1, 28, 28]
 *   y: [batch] (labels as bytes)
 */
@Serializable
public data class FashionMNISTDataset(
    val images: List<FashionMNISTImage>,
    @Transient private val executionContext: ExecutionContext = DefaultDataExecutionContext()
) : Dataset<FashionMNISTImage, Float>() {

    override val xSize: Int get() = images.size

    override fun getX(idx: Int): FashionMNISTImage = images[idx]

    override fun getY(idx: Int): Float = images[idx].label.toInt().toFloat()

    override fun shuffle(): Dataset<FashionMNISTImage, Float> {
        val shuffled = images.toMutableList()
        shuffled.shuffle(Random.Default)
        return FashionMNISTDataset(shuffled, executionContext)
    }

    override fun split(splitRatio: Double): Pair<Dataset<FashionMNISTImage, Float>, Dataset<FashionMNISTImage, Float>> {
        require(splitRatio > 0.0 && splitRatio < 1.0) { "splitRatio must be in (0,1)" }
        val splitIndex = (images.size * splitRatio).toInt()
        val first = images.subList(0, splitIndex)
        val second = images.subList(splitIndex, images.size)
        return FashionMNISTDataset(first, executionContext) to FashionMNISTDataset(second, executionContext)
    }

    /**
     * Creates a DataBatch with memory-efficient Int8 tensors from raw byte arrays.
     */
    override fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V> {
        val actualLen = min(batchLength, xSize - batchStart)
        val batchImages = images.subList(batchStart, batchStart + actualLen)

        // Concatenate raw image bytes (no normalization) for memory efficiency
        val xData = ByteArray(actualLen * FashionMNISTConstants.IMAGE_PIXELS)
        var offset = 0
        for (sample in batchImages) {
            val bytes = sample.image
            bytes.copyInto(xData, destinationOffset = offset, startIndex = 0, endIndex = FashionMNISTConstants.IMAGE_PIXELS)
            offset += FashionMNISTConstants.IMAGE_PIXELS
        }

        // Shape as [batch, 1, 28, 28]
        val xShape = Shape(actualLen, 1, FashionMNISTConstants.IMAGE_SIZE, FashionMNISTConstants.IMAGE_SIZE)
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
    public fun subset(fromIndex: Int, toIndex: Int): FashionMNISTDataset {
        return FashionMNISTDataset(images.subList(fromIndex, toIndex), executionContext)
    }
}

/**
 * Configuration for the Fashion-MNIST loader.
 *
 * @property cacheDir The directory where downloaded files will be cached.
 * @property useCache Whether to use cached files if available.
 */
public data class FashionMNISTLoaderConfig(
    val cacheDir: String = "fashion-mnist-data",
    val useCache: Boolean = true,
    val trainImagesUri: String = FashionMNISTConstants.TRAIN_IMAGES_URL,
    val trainLabelsUri: String = FashionMNISTConstants.TRAIN_LABELS_URL,
    val testImagesUri: String = FashionMNISTConstants.TEST_IMAGES_URL,
    val testLabelsUri: String = FashionMNISTConstants.TEST_LABELS_URL
)

/**
 * Constants for the Fashion-MNIST dataset.
 */
public object FashionMNISTConstants {
    public const val IMAGE_SIZE: Int = 28
    public const val IMAGE_PIXELS: Int = IMAGE_SIZE * IMAGE_SIZE

    public const val TRAIN_IMAGES_FILENAME: String = "train-images-idx3-ubyte.gz"
    public const val TRAIN_LABELS_FILENAME: String = "train-labels-idx1-ubyte.gz"
    public const val TEST_IMAGES_FILENAME: String = "t10k-images-idx3-ubyte.gz"
    public const val TEST_LABELS_FILENAME: String = "t10k-labels-idx1-ubyte.gz"

    // Fashion-MNIST URLs from the official GitHub repository
    public const val TRAIN_IMAGES_URL: String =
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
    public const val TRAIN_LABELS_URL: String =
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
    public const val TEST_IMAGES_URL: String =
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
    public const val TEST_LABELS_URL: String =
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

    /**
     * Class names for Fashion-MNIST labels.
     */
    public val CLASS_NAMES: List<String> = listOf(
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    )
}

/**
 * Interface for the Fashion-MNIST loader.
 */
public interface FashionMNISTLoader {
    /**
     * Loads the Fashion-MNIST training dataset.
     *
     * @return The Fashion-MNIST training dataset.
     */
    public suspend fun loadTrainingData(): FashionMNISTDataset

    /**
     * Loads the Fashion-MNIST test dataset.
     *
     * @return The Fashion-MNIST test dataset.
     */
    public suspend fun loadTestData(): FashionMNISTDataset
}
