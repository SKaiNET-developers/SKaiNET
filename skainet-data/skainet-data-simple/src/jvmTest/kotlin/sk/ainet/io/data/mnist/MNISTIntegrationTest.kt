package sk.ainet.io.data.mnist

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.data.mnist.MNISTConstants
import sk.ainet.data.mnist.MNISTLoaderConfig
import sk.ainet.data.mnist.MNISTLoaderFactory
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.Int8
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Simple integration test that:
 * 1) Downloads MNIST dataset (ignores cache)
 * 2) Creates a train/test split from the training data
 * 3) Selects one random image and converts it to a tensor
 */
class MNISTIntegrationTest {

    @Test
    fun downloadSplitAndTensorize() = runBlocking {
        // always ignore cache to force download as requested
        val config = MNISTLoaderConfig(
            cacheDir = kotlin.io.path.createTempDirectory(prefix = "mnist-it-").toFile().absolutePath,
            useCache = false
        )

        val loader = MNISTLoaderFactory.create(config)

        // 1) Download training dataset
        val trainDataset = loader.loadTrainingData()
        assertTrue(trainDataset.images.isNotEmpty(), "Training dataset should not be empty")

        // 2) Create a train/test split (80/20)
        val (trainSplit, testSplit) = trainDataset.shuffle().split(0.8)
        assertTrue(trainSplit.xSize > 0, "Train split should not be empty")
        assertTrue(testSplit.xSize > 0, "Test split should not be empty")

        // 3) Pick a random image from the train split and convert to a tensor
        val idx = Random(1234).nextInt(trainSplit.xSize)
        val sample = trainSplit.getX(idx)

        // Expect raw bytes of size 28x28
        assertEquals(MNISTConstants.IMAGE_PIXELS, sample.image.size)

        // Convert to an Int8 tensor with shape [1, 28, 28]
        val ctx = DefaultDataExecutionContext()
        val shape = Shape(1, MNISTConstants.IMAGE_SIZE, MNISTConstants.IMAGE_SIZE)
        val tensor = ctx.fromByteArray<Int8, Byte>(shape, Int8::class, sample.image)

        assertEquals(shape, tensor.shape)

        // Basic sanity checks on values range and size
        assertEquals(MNISTConstants.IMAGE_PIXELS, tensor.volume)

        // Verify tensor isn't all zeros by comparing with original bytes
        // and also spot-check a few indices match exactly
        var nonZeroCount = 0
        val h = MNISTConstants.IMAGE_SIZE
        val w = MNISTConstants.IMAGE_SIZE
        for (yy in 0 until h) {
            for (xx in 0 until w) {
                val tVal: Byte = tensor.data[0, yy, xx]
                val bVal: Byte = sample.image[yy * w + xx]
                if (tVal.toInt() != 0) nonZeroCount++
                // spot check: values match source bytes
                if ((yy == 0 && (xx == 0 || xx == w - 1)) ||
                    (yy == h - 1 && (xx == 0 || xx == w - 1)) ||
                    (yy == h / 2 && xx == w / 2)) {
                    assertEquals(bVal, tVal, "Tensor value should match source byte at ($yy,$xx)")
                }
            }
        }

        // At least one pixel in MNIST image should be non-zero
        assertTrue(nonZeroCount > 0, "Constructed tensor appears to be all zeros — expected some non-zero pixels")
    }
}
