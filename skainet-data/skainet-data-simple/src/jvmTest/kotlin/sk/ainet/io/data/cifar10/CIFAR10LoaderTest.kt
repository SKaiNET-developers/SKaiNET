package sk.ainet.io.data.cifar10

import kotlinx.coroutines.runBlocking
import sk.ainet.data.cifar10.CIFAR10Constants
import sk.ainet.data.cifar10.CIFAR10Image
import sk.ainet.data.cifar10.CIFAR10LoaderConfig
import sk.ainet.data.cifar10.CIFAR10LoaderCommon
import sk.ainet.data.cifar10.createCIFAR10Loader
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CIFAR10LoaderTest {

    @Test
    fun testLoadTrainingData() = runBlocking {
        val loader = createFakeLoader()

        val dataset = loader.loadTrainingData()

        // 5 batches * 3 images per batch = 15 images in our fake data
        assertEquals(15, dataset.images.size)
        val firstImage = dataset.images.first()
        assertEquals(CIFAR10Constants.IMAGE_BYTES, firstImage.image.size)
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
        assertEquals(CIFAR10Constants.IMAGE_BYTES, firstImage.image.size)

        val cachedDataset = loader.loadTestData()
        assertEquals(dataset.images, cachedDataset.images)
    }

    @Test
    fun testDatasetSubset() = runBlocking {
        val loader = createFakeLoader()
        val dataset = loader.loadTestData()

        val subset = dataset.subset(0, 2)

        assertEquals(2, subset.images.size)
        assertEquals(dataset.images[0], subset.images[0])
        assertEquals(dataset.images[1], subset.images[1])
    }

    @Test
    fun testShuffledDatasetViewCanCreateBatch() = runBlocking {
        val dataset = createFakeLoader().loadTrainingData()
        val shuffled = dataset.shuffle(seed = 123)

        val batch = shuffled.batchIterator<Int8, Byte>(4).next()

        assertEquals(4, batch.batchSize)
        assertEquals(4, batch.indices.size)
        assertEquals(4, batch.x[0].shape[0])
        assertEquals(4, batch.y.shape[0])
    }

    @Test
    fun testLoaderConfiguration() {
        val config = CIFAR10LoaderConfig(
            cacheDir = "custom-cache-dir",
            useCache = false,
            archiveUri = "hf+https://huggingface.co/datasets/cifar10/resolve/main/cifar-10-binary.tar.gz"
        )
        val loader = createCIFAR10Loader(config)

        assertNotNull(loader)
    }

    @Test
    fun testClassNames() {
        assertEquals("airplane", CIFAR10Constants.CLASS_NAMES[0])
        assertEquals("automobile", CIFAR10Constants.CLASS_NAMES[1])
        assertEquals("bird", CIFAR10Constants.CLASS_NAMES[2])
        assertEquals("cat", CIFAR10Constants.CLASS_NAMES[3])
        assertEquals("deer", CIFAR10Constants.CLASS_NAMES[4])
        assertEquals("dog", CIFAR10Constants.CLASS_NAMES[5])
        assertEquals("frog", CIFAR10Constants.CLASS_NAMES[6])
        assertEquals("horse", CIFAR10Constants.CLASS_NAMES[7])
        assertEquals("ship", CIFAR10Constants.CLASS_NAMES[8])
        assertEquals("truck", CIFAR10Constants.CLASS_NAMES[9])
    }

    @Test
    fun testImageClassName() {
        val image = CIFAR10Image(ByteArray(CIFAR10Constants.IMAGE_BYTES), 3)
        assertEquals("cat", image.className())
    }

    @Test
    fun testImageDimensions() {
        assertEquals(32, CIFAR10Constants.IMAGE_SIZE)
        assertEquals(3, CIFAR10Constants.NUM_CHANNELS)
        assertEquals(3072, CIFAR10Constants.IMAGE_BYTES)  // 3 * 32 * 32
    }
}

private fun createFakeLoader(config: CIFAR10LoaderConfig = CIFAR10LoaderConfig()): FakeCIFAR10Loader {
    return FakeCIFAR10Loader(config)
}

private class FakeCIFAR10Loader(
    config: CIFAR10LoaderConfig
) : CIFAR10LoaderCommon(config) {

    override suspend fun downloadAndExtractBatch(batchFilename: String): ByteArray {
        // Generate fake batch data based on filename
        val images = when (batchFilename) {
            CIFAR10Constants.DATA_BATCH_1_FILENAME -> EXPECTED_BATCH_1_DATA
            CIFAR10Constants.DATA_BATCH_2_FILENAME -> EXPECTED_BATCH_2_DATA
            CIFAR10Constants.DATA_BATCH_3_FILENAME -> EXPECTED_BATCH_3_DATA
            CIFAR10Constants.DATA_BATCH_4_FILENAME -> EXPECTED_BATCH_4_DATA
            CIFAR10Constants.DATA_BATCH_5_FILENAME -> EXPECTED_BATCH_5_DATA
            CIFAR10Constants.TEST_BATCH_FILENAME -> EXPECTED_TEST_DATA
            else -> error("Unexpected filename $batchFilename")
        }
        return buildBatchFile(images)
    }
}

// Each training batch has 3 fake images
private val EXPECTED_BATCH_1_DATA = listOf(
    sampleCifar10Image(seed = 0, label = 0),   // airplane
    sampleCifar10Image(seed = 1, label = 1),   // automobile
    sampleCifar10Image(seed = 2, label = 2)    // bird
)

private val EXPECTED_BATCH_2_DATA = listOf(
    sampleCifar10Image(seed = 10, label = 3),  // cat
    sampleCifar10Image(seed = 11, label = 4),  // deer
    sampleCifar10Image(seed = 12, label = 5)   // dog
)

private val EXPECTED_BATCH_3_DATA = listOf(
    sampleCifar10Image(seed = 20, label = 6),  // frog
    sampleCifar10Image(seed = 21, label = 7),  // horse
    sampleCifar10Image(seed = 22, label = 8)   // ship
)

private val EXPECTED_BATCH_4_DATA = listOf(
    sampleCifar10Image(seed = 30, label = 9),  // truck
    sampleCifar10Image(seed = 31, label = 0),  // airplane
    sampleCifar10Image(seed = 32, label = 1)   // automobile
)

private val EXPECTED_BATCH_5_DATA = listOf(
    sampleCifar10Image(seed = 40, label = 2),  // bird
    sampleCifar10Image(seed = 41, label = 3),  // cat
    sampleCifar10Image(seed = 42, label = 4)   // deer
)

private val EXPECTED_TEST_DATA = listOf(
    sampleCifar10Image(seed = 100, label = 5), // dog
    sampleCifar10Image(seed = 101, label = 6)  // frog
)

private fun sampleCifar10Image(seed: Int, label: Int): CIFAR10Image {
    val pixels = ByteArray(CIFAR10Constants.IMAGE_BYTES) { idx ->
        ((seed + idx) % 256).toByte()
    }
    return CIFAR10Image(pixels, label.toByte())
}

/**
 * Build a CIFAR-10 batch file format:
 * Each image: 1 byte label + 3072 bytes pixel data
 */
private fun buildBatchFile(images: List<CIFAR10Image>): ByteArray {
    val bytesPerImage = 1 + CIFAR10Constants.IMAGE_BYTES
    val buffer = ByteArray(images.size * bytesPerImage)

    var offset = 0
    for (image in images) {
        buffer[offset] = image.label
        image.image.copyInto(buffer, destinationOffset = offset + 1)
        offset += bytesPerImage
    }
    return buffer
}
