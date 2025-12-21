package de.jugda.knanogpt.core.data


import de.jugda.knanogpt.core.tensor.TrainTestSplitter
import sk.ainet.apps.knanogpt.data.BatchProvider
import sk.ainet.apps.knanogpt.data.ResourcesDataProvider
import kotlin.test.Test
import kotlin.test.assertEquals


class BatchProviderTest {

    @Test
    fun `get back delivers test`() {
        val provider = ResourcesDataProvider("input.txt")
        val data = provider.load()
        val splitter = TrainTestSplitter(data)
        val (trainData, testData) = splitter.split(0.9f)
        assertEquals(data.shape.volume, 1003854 + 111540)
        assertEquals(trainData.shape.volume, 1003854)
        assertEquals(testData.shape.volume, 111540)

        val batch = BatchProvider(trainData, testData, 8, 4)
        val (x, y) = batch.getRandomBatch(true)
        assertEquals(x.shape.volume, 32)
        assertEquals(y.shape.volume, 32)
        assertEquals(x.shape.dimensions.size, 2)
        assertEquals(y.shape.dimensions.size, 2)
        assertEquals(x.shape.dimensions[0], 4)
        assertEquals(y.shape.dimensions[1], 8)
        assertEquals(y.shape.dimensions[0], 4)
        assertEquals(y.shape.dimensions[1], 8)
    }
}