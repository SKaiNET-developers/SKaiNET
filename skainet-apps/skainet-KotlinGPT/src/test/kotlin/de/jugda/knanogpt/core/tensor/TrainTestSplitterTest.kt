package de.jugda.knanogpt.core.tensor

import sk.ainet.apps.knanogpt.data.ResourcesDataProvider
import kotlin.test.Test
import kotlin.test.assertEquals


class TrainTestSplitterTest {

    @Test
    fun `train test split test`() {
        val provider = ResourcesDataProvider("input.txt")
        val tensor = provider.load()
        val splitter = TrainTestSplitter(tensor)
        val (train, test) = splitter.split(0.9f)
        assertEquals(train.shape.volume, (tensor.shape.volume * 0.9).toInt())
    }
}