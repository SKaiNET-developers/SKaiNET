package sk.ainet.io.data.fashionmnist

import kotlinx.coroutines.runBlocking
import sk.ainet.data.fashionmnist.FashionMNISTConstants
import sk.ainet.data.fashionmnist.FashionMNISTImage
import sk.ainet.data.fashionmnist.FashionMNISTLoaderConfig
import sk.ainet.data.fashionmnist.FashionMNISTLoaderCommon
import sk.ainet.data.fashionmnist.createFashionMNISTLoader
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class FashionMNISTLoaderTest {

    @Test
    fun testLoadTrainingData() = runBlocking {
        val loader = createFakeLoader()

        val dataset = loader.loadTrainingData()

        assertEquals(EXPECTED_TRAINING_DATA.size, dataset.images.size)
        assertEquals(EXPECTED_TRAINING_DATA, dataset.images)
        val firstImage = dataset.images.first()
        assertEquals(FashionMNISTConstants.IMAGE_PIXELS, firstImage.image.size)
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
        assertEquals(FashionMNISTConstants.IMAGE_PIXELS, firstImage.image.size)

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
    fun testShuffledDatasetViewCanCreateBatch() = runBlocking {
        val dataset = createFakeLoader().loadTrainingData()
        val shuffled = dataset.shuffle(seed = 123)

        val batch = shuffled.batchIterator<Int8, Byte>(2).next()

        assertEquals(2, batch.batchSize)
        assertEquals(2, batch.indices.size)
        assertEquals(2, batch.x[0].shape[0])
        assertEquals(2, batch.y.shape[0])
    }

    @Test
    fun testLoaderConfiguration() {
        val config = FashionMNISTLoaderConfig(
            cacheDir = "custom-cache-dir",
            useCache = false,
            trainImagesUri = "file:///datasets/fashion-mnist/train-images",
            trainLabelsUri = "hf+https://huggingface.co/datasets/zalando-datasets/fashion_mnist/resolve/main/train-labels"
        )
        val loader = createFashionMNISTLoader(config)

        assertNotNull(loader)
    }

    @Test
    fun testClassNames() {
        assertEquals("T-shirt/top", FashionMNISTConstants.CLASS_NAMES[0])
        assertEquals("Trouser", FashionMNISTConstants.CLASS_NAMES[1])
        assertEquals("Pullover", FashionMNISTConstants.CLASS_NAMES[2])
        assertEquals("Dress", FashionMNISTConstants.CLASS_NAMES[3])
        assertEquals("Coat", FashionMNISTConstants.CLASS_NAMES[4])
        assertEquals("Sandal", FashionMNISTConstants.CLASS_NAMES[5])
        assertEquals("Shirt", FashionMNISTConstants.CLASS_NAMES[6])
        assertEquals("Sneaker", FashionMNISTConstants.CLASS_NAMES[7])
        assertEquals("Bag", FashionMNISTConstants.CLASS_NAMES[8])
        assertEquals("Ankle boot", FashionMNISTConstants.CLASS_NAMES[9])
    }

    @Test
    fun testImageClassName() {
        val image = FashionMNISTImage(ByteArray(FashionMNISTConstants.IMAGE_PIXELS), 3)
        assertEquals("Dress", image.className())
    }
}

private fun createFakeLoader(config: FashionMNISTLoaderConfig = FashionMNISTLoaderConfig()): FakeFashionMNISTLoader {
    return FakeFashionMNISTLoader(
        config = config,
        trainingImagesBytes = TRAINING_IMAGES_BYTES,
        trainingLabelsBytes = TRAINING_LABELS_BYTES,
        testImagesBytes = TEST_IMAGES_BYTES,
        testLabelsBytes = TEST_LABELS_BYTES
    )
}

private class FakeFashionMNISTLoader(
    config: FashionMNISTLoaderConfig,
    private val trainingImagesBytes: ByteArray,
    private val trainingLabelsBytes: ByteArray,
    private val testImagesBytes: ByteArray,
    private val testLabelsBytes: ByteArray
) : FashionMNISTLoaderCommon(config) {
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray {
        val data = when (filename) {
            FashionMNISTConstants.TRAIN_IMAGES_FILENAME -> trainingImagesBytes
            FashionMNISTConstants.TRAIN_LABELS_FILENAME -> trainingLabelsBytes
            FashionMNISTConstants.TEST_IMAGES_FILENAME -> testImagesBytes
            FashionMNISTConstants.TEST_LABELS_FILENAME -> testLabelsBytes
            else -> error("Unexpected filename $filename")
        }
        return data.copyOf()
    }
}

private val EXPECTED_TRAINING_DATA = listOf(
    sampleFashionMnistImage(seed = 0, label = 3),  // Dress
    sampleFashionMnistImage(seed = 1, label = 7),  // Sneaker
    sampleFashionMnistImage(seed = 2, label = 1)   // Trouser
)

private val EXPECTED_TEST_DATA = listOf(
    sampleFashionMnistImage(seed = 10, label = 2),  // Pullover
    sampleFashionMnistImage(seed = 11, label = 4)   // Coat
)

private val TRAINING_IMAGES_BYTES = buildImagesFile(EXPECTED_TRAINING_DATA.map { it.image })
private val TRAINING_LABELS_BYTES = buildLabelsFile(EXPECTED_TRAINING_DATA.map { it.label })
private val TEST_IMAGES_BYTES = buildImagesFile(EXPECTED_TEST_DATA.map { it.image })
private val TEST_LABELS_BYTES = buildLabelsFile(EXPECTED_TEST_DATA.map { it.label })

private fun sampleFashionMnistImage(seed: Int, label: Int): FashionMNISTImage {
    val pixels = ByteArray(FashionMNISTConstants.IMAGE_PIXELS) { idx ->
        ((seed + idx) % 256).toByte()
    }
    return FashionMNISTImage(pixels, label.toByte())
}

private fun buildImagesFile(images: List<ByteArray>): ByteArray {
    require(images.isNotEmpty()) { "images must not be empty" }
    images.forEach { require(it.size == FashionMNISTConstants.IMAGE_PIXELS) }

    val headerSize = 16
    val buffer = ByteArray(headerSize + images.size * FashionMNISTConstants.IMAGE_PIXELS)
    buffer.writeInt(headerOffset = 0, value = 2051)
    buffer.writeInt(headerOffset = 4, value = images.size)
    buffer.writeInt(headerOffset = 8, value = FashionMNISTConstants.IMAGE_SIZE)
    buffer.writeInt(headerOffset = 12, value = FashionMNISTConstants.IMAGE_SIZE)

    var offset = headerSize
    for (image in images) {
        image.copyInto(buffer, destinationOffset = offset)
        offset += FashionMNISTConstants.IMAGE_PIXELS
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
