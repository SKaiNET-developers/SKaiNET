package de.jugda.knanogpt.core.tensor

import sk.ainet.apps.knanogpt.data.ResourcesDataProvider
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals


class FactoriesTest {

    @Test
    fun testRandInt() {
        val provider = ResourcesDataProvider("input.txt")
        val tensor = provider.load()

        val dataLen = tensor.elements.size
        val blockSize = 8
        val batchSize = 4 // # how many independent sequences will we process in parallel?
        val random = Random(1337)
        val randtensor = randint(random, 0, dataLen - blockSize, Shape(batchSize))
        assertContentEquals(
            randtensor.elements,
            Tensor(Shape(batchSize), listOf(139827.0, 146480.0, 314577.0, 99015.0).toDoubleArray()).elements
        )
    }

}