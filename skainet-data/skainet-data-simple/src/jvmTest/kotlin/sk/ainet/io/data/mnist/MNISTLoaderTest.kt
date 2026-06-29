package sk.ainet.io.data.mnist

import kotlinx.coroutines.runBlocking
import sk.ainet.data.mnist.MNISTConstants
import sk.ainet.data.mnist.MNISTImage
import sk.ainet.data.mnist.MNISTLoaderConfig
import sk.ainet.data.mnist.MNISTLoaderFactory
import sk.ainet.data.mnist.MNISTLoaderCommon
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class MNISTLoaderTest {

    @Test
    fun testLoadTrainingData() = runBlocking {
        val loader = createFakeLoader()

        val dataset = loader.loadTrainingData()

        assertEquals(EXPECTED_TRAINING_DATA.size, dataset.images.size)
        assertEquals(EXPECTED_TRAINING_DATA, dataset.images)
        val firstImage = dataset.images.first()
        assertEquals(MNISTConstants.IMAGE_PIXELS, firstImage.image.size)
        assertTrue(firstImage.label in 0..9)

        val cachedDataset = loader.loadTrainingData()
        assertEquals(dataset.images, cachedDataset.images)
    }

    @Test
    fun testLoadTestData() = runBlocking {
        val loader = createFakeLoader()

        val dataset = loader.loadTestData()

        assertEquals(EXPECTED_TEST_DATA.size, dataset.images.size)
        assertEquals(EXPECTED_TEST_DATA, dataset.images)
        val firstImage = dataset.images.first()
        assertEquals(MNISTConstants.IMAGE_PIXELS, firstImage.image.size)

        val cachedDataset = loader.loadTestData()
        assertEquals(dataset.images, cachedDataset.images)
    }

    @Test
    fun testDatasetSubset() = runBlocking {
        val loader = createFakeLoader()
        val dataset = loader.loadTrainingData()

        val subset = dataset.subset(0, 2)

        assertEquals(2, subset.images.size)
        assertEquals(dataset.images[0], subset.images[0])
        assertEquals(dataset.images[1], subset.images[1])
    }

    @Test
    fun testLoaderConfiguration() {
        val config = MNISTLoaderConfig(
            cacheDir = "custom-cache-dir",
            useCache = false,
            trainImagesUri = "file:///datasets/mnist/train-images",
            trainLabelsUri = "hf+https://huggingface.co/datasets/mnist/mnist/resolve/main/train-labels"
        )
        val loader = MNISTLoaderFactory.create(config)

        assertNotNull(loader)
    }

    @Test
    fun testJvmLoaderReadsConfiguredFileUris() = runBlocking {
        val root = Files.createTempDirectory("skainet-mnist-loader-test").toFile()
        try {
            val trainImages = root.resolve("train-images.idx")
            val trainLabels = root.resolve("train-labels.idx")
            trainImages.writeBytes(TRAINING_IMAGES_BYTES)
            trainLabels.writeBytes(TRAINING_LABELS_BYTES)
            val config = MNISTLoaderConfig(
                cacheDir = root.resolve("cache").absolutePath,
                useCache = false,
                trainImagesUri = trainImages.toURI().toString(),
                trainLabelsUri = trainLabels.toURI().toString()
            )

            val dataset = MNISTLoaderFactory.create(config).loadTrainingData()

            assertEquals(EXPECTED_TRAINING_DATA, dataset.images)
        } finally {
            root.deleteRecursively()
        }
    }
}

private fun createFakeLoader(config: MNISTLoaderConfig = MNISTLoaderConfig()): FakeMNISTLoader {
    return FakeMNISTLoader(
        config = config,
        trainingImagesBytes = TRAINING_IMAGES_BYTES,
        trainingLabelsBytes = TRAINING_LABELS_BYTES,
        testImagesBytes = TEST_IMAGES_BYTES,
        testLabelsBytes = TEST_LABELS_BYTES
    )
}

private class FakeMNISTLoader(
    config: MNISTLoaderConfig,
    private val trainingImagesBytes: ByteArray,
    private val trainingLabelsBytes: ByteArray,
    private val testImagesBytes: ByteArray,
    private val testLabelsBytes: ByteArray
) : MNISTLoaderCommon(config) {
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray {
        val data = when (filename) {
            MNISTConstants.TRAIN_IMAGES_FILENAME -> trainingImagesBytes
            MNISTConstants.TRAIN_LABELS_FILENAME -> trainingLabelsBytes
            MNISTConstants.TEST_IMAGES_FILENAME -> testImagesBytes
            MNISTConstants.TEST_LABELS_FILENAME -> testLabelsBytes
            else -> error("Unexpected filename $filename")
        }
        return data.copyOf()
    }
}

private val EXPECTED_TRAINING_DATA = listOf(
    sampleMnistImage(seed = 0, label = 3),
    sampleMnistImage(seed = 1, label = 7),
    sampleMnistImage(seed = 2, label = 1)
)

private val EXPECTED_TEST_DATA = listOf(
    sampleMnistImage(seed = 10, label = 2),
    sampleMnistImage(seed = 11, label = 4)
)

private val TRAINING_IMAGES_BYTES = buildImagesFile(EXPECTED_TRAINING_DATA.map { it.image })
private val TRAINING_LABELS_BYTES = buildLabelsFile(EXPECTED_TRAINING_DATA.map { it.label })
private val TEST_IMAGES_BYTES = buildImagesFile(EXPECTED_TEST_DATA.map { it.image })
private val TEST_LABELS_BYTES = buildLabelsFile(EXPECTED_TEST_DATA.map { it.label })

private fun sampleMnistImage(seed: Int, label: Int): MNISTImage {
    val pixels = ByteArray(MNISTConstants.IMAGE_PIXELS) { idx ->
        ((seed + idx) % 256).toByte()
    }
    return MNISTImage(pixels, label.toByte())
}

private fun buildImagesFile(images: List<ByteArray>): ByteArray {
    require(images.isNotEmpty()) { "images must not be empty" }
    images.forEach { require(it.size == MNISTConstants.IMAGE_PIXELS) }

    val headerSize = 16
    val buffer = ByteArray(headerSize + images.size * MNISTConstants.IMAGE_PIXELS)
    buffer.writeInt(headerOffset = 0, value = 2051)
    buffer.writeInt(headerOffset = 4, value = images.size)
    buffer.writeInt(headerOffset = 8, value = MNISTConstants.IMAGE_SIZE)
    buffer.writeInt(headerOffset = 12, value = MNISTConstants.IMAGE_SIZE)

    var offset = headerSize
    for (image in images) {
        image.copyInto(buffer, destinationOffset = offset)
        offset += MNISTConstants.IMAGE_PIXELS
    }
    return buffer
}

private fun buildLabelsFile(labels: List<Byte>): ByteArray {
    val headerSize = 8
    val buffer = ByteArray(headerSize + labels.size)
    buffer.writeInt(headerOffset = 0, value = 2049)
    buffer.writeInt(headerOffset = 4, value = labels.size)
    labels.forEachIndexed { index, byte ->
        buffer[headerSize + index] = byte
    }
    return buffer
}

private fun ByteArray.writeInt(headerOffset: Int, value: Int) {
    this[headerOffset] = (value ushr 24).toByte()
    this[headerOffset + 1] = (value ushr 16).toByte()
    this[headerOffset + 2] = (value ushr 8).toByte()
    this[headerOffset + 3] = value.toByte()
}
