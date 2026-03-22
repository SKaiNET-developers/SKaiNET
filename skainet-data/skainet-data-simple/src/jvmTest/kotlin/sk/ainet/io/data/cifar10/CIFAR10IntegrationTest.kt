package sk.ainet.io.data.cifar10

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.data.cifar10.CIFAR10Constants
import sk.ainet.data.cifar10.CIFAR10LoaderConfig
import sk.ainet.data.cifar10.createCIFAR10Loader
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.Int8
import kotlin.random.Random
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Integration test that:
 * 1) Downloads CIFAR-10 dataset (ignores cache)
 * 2) Creates a train/test split from the training data
 * 3) Selects one random image and converts it to a tensor
 *
 * Note: This test downloads ~170MB of data and may take several minutes.
 */
class CIFAR10IntegrationTest {

    @Ignore("Integration test — downloads ~170 MB; run manually")
    @Test
    fun downloadSplitAndTensorize() = runBlocking {
        // always ignore cache to force download as requested
        val config = CIFAR10LoaderConfig(
            cacheDir = kotlin.io.path.createTempDirectory(prefix = "cifar10-it-").toFile().absolutePath,
            useCache = false
        )

        val loader = createCIFAR10Loader(config)

        // 1) Download training dataset
        val trainDataset = loader.loadTrainingData()
        assertTrue(trainDataset.images.isNotEmpty(), "Training dataset should not be empty")
        assertEquals(CIFAR10Constants.NUM_TRAINING_IMAGES, trainDataset.images.size,
            "Training dataset should have ${CIFAR10Constants.NUM_TRAINING_IMAGES} images")

        // 2) Create a train/test split (80/20)
        val (trainSplit, testSplit) = trainDataset.shuffle().split(0.8)
        assertTrue(trainSplit.xSize > 0, "Train split should not be empty")
        assertTrue(testSplit.xSize > 0, "Test split should not be empty")

        // 3) Pick a random image from the train split and convert to a tensor
        val idx = Random(1234).nextInt(trainSplit.xSize)
        val sample = trainSplit.getX(idx)

        // Expect raw bytes of size 3*32*32 = 3072
        assertEquals(CIFAR10Constants.IMAGE_BYTES, sample.image.size)

        // Verify label is in valid range and has a class name
        assertTrue(sample.label in 0..9, "Label should be between 0 and 9")
        assertTrue(sample.className() in CIFAR10Constants.CLASS_NAMES, "Class name should be valid")

        // Convert to an Int8 tensor with shape [1, 3, 32, 32]
        val ctx = DefaultDataExecutionContext()
        val shape = Shape(1, CIFAR10Constants.NUM_CHANNELS, CIFAR10Constants.IMAGE_SIZE, CIFAR10Constants.IMAGE_SIZE)
        val tensor = ctx.fromByteArray<Int8, Byte>(shape, Int8::class, sample.image)

        assertEquals(shape, tensor.shape)

        // Basic sanity checks on values range and size
        assertEquals(CIFAR10Constants.IMAGE_BYTES, tensor.volume)

        // Verify tensor isn't all zeros
        var nonZeroCount = 0
        val c = CIFAR10Constants.NUM_CHANNELS
        val h = CIFAR10Constants.IMAGE_SIZE
        val w = CIFAR10Constants.IMAGE_SIZE
        for (ch in 0 until c) {
            for (yy in 0 until h) {
                for (xx in 0 until w) {
                    val tVal: Byte = tensor.data[0, ch, yy, xx]
                    if (tVal.toInt() != 0) nonZeroCount++
                }
            }
        }

        // CIFAR-10 images are real photos, should have many non-zero pixels
        assertTrue(nonZeroCount > 0, "Constructed tensor appears to be all zeros — expected some non-zero pixels")
    }

    @Ignore("Integration test — downloads ~170 MB; run manually")
    @Test
    fun downloadTestDataset() = runBlocking {
        val config = CIFAR10LoaderConfig(
            cacheDir = kotlin.io.path.createTempDirectory(prefix = "cifar10-it-test-").toFile().absolutePath,
            useCache = false
        )

        val loader = createCIFAR10Loader(config)

        // Download test dataset
        val testDataset = loader.loadTestData()
        assertTrue(testDataset.images.isNotEmpty(), "Test dataset should not be empty")
        assertEquals(CIFAR10Constants.NUM_TEST_IMAGES, testDataset.images.size,
            "Test dataset should have ${CIFAR10Constants.NUM_TEST_IMAGES} images")

        // Verify first image
        val firstImage = testDataset.getX(0)
        assertEquals(CIFAR10Constants.IMAGE_BYTES, firstImage.image.size)
        assertTrue(firstImage.label in 0..9)
    }
}
