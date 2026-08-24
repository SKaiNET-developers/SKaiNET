package sk.ainet.data.iris

import kotlinx.coroutines.runBlocking
import sk.ainet.data.Dataset
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class IrisDatasetTest {

    @Test
    fun loadReturnsExactly150SamplesWith50PerClass() = runBlocking {
        val dataset = Iris.load()

        assertEquals(150, dataset.xSize)
        assertEquals(
            mapOf(0 to 50, 1 to 50, 2 to 50),
            classCounts(dataset)
        )
    }

    @Test
    fun featuresStayWithinDocumentedRanges() = runBlocking {
        val dataset = Iris.load()

        for (idx in 0 until dataset.xSize) {
            val x = dataset.getX(idx)
            assertEquals(4, x.size)
            assertTrue(x[0] in 4.3f..7.9f, "sepalLength out of range at $idx: ${x[0]}")
            assertTrue(x[1] in 2.0f..4.4f, "sepalWidth out of range at $idx: ${x[1]}")
            assertTrue(x[2] in 1.0f..6.9f, "petalLength out of range at $idx: ${x[2]}")
            assertTrue(x[3] in 0.1f..2.5f, "petalWidth out of range at $idx: ${x[3]}")
        }
    }

    @Test
    fun classIndicesAgreeWithClassNamesOrdering() = runBlocking {
        val dataset = Iris.load()

        // The embedded CSV is grouped in blocks of 50 per species.
        for ((blockStart, expectedLabel) in listOf(0 to 0, 50 to 1, 100 to 2)) {
            val sample = dataset.samples[blockStart]
            assertEquals(Iris.classNames[expectedLabel], speciesOfCsvLine(blockStart))
            assertEquals(expectedLabel, sample.label)
            assertEquals(expectedLabel, dataset.getY(blockStart))
        }

        // Known-good first row of the canonical dataset.
        assertEquals(IrisSample(5.1f, 3.5f, 1.4f, 0.2f, 0), dataset.samples.first())
    }

    @Test
    fun dataBatchProducesFeatureAndOneHotTensorsOfTheRightShape() = runBlocking {
        val dataset = Iris.load()
        val batch = dataset.dataBatch<FP32, Float>(batchStart = 16, batchLength = 8)

        assertEquals(8, batch.batchSize)
        assertEquals(listOf(8, 4), batch.x[0].shape.dimensions.toList())
        assertEquals(listOf(8, 3), batch.y.shape.dimensions.toList())
        assertEquals((16 until 24).toList(), batch.indices.toList())

        val x = batch.x[0].data.copyToFloatArray()
        val y = batch.y.data.copyToFloatArray()
        for (row in 0 until 8) {
            val sourceIdx = 16 + row
            val features = dataset.getX(sourceIdx)
            for (col in 0 until 4) {
                assertEquals(features[col], x[row * 4 + col], "x mismatch at row $row column $col")
            }
            assertEquals(1.0f, y[row * 3 + dataset.getY(sourceIdx)])
            assertEquals(
                1.0f,
                y[row * 3] + y[row * 3 + 1] + y[row * 3 + 2],
                "one-hot row $row must sum to 1.0"
            )
        }
    }

    @Test
    fun stratifiedSplitGives120Train30TestWithBalancedClasses() = runBlocking {
        val (train, test) = Iris.load().split(splitRatio = 0.8, seed = 42L, stratified = true)

        assertEquals(120, train.xSize)
        assertEquals(30, test.xSize)
        assertEquals(mapOf(0 to 40, 1 to 40, 2 to 40), classCounts(train))
        assertEquals(mapOf(0 to 10, 1 to 10, 2 to 10), classCounts(test))
    }

    @Test
    fun splitThenBatchDoesNotThrow() = runBlocking {
        val (train, _) = Iris.load().split(splitRatio = 0.8, seed = 42L, stratified = true)

        // split() returns a non-contiguous index view; batching over that view
        // must route through createIndexedDataBatch instead of failing.
        val iterator = train.batchIterator<FP32, Float>(batchSize = 16)
        var rows = 0
        while (iterator.hasNext()) {
            rows += iterator.next().batchSize
        }
        assertEquals(train.xSize, rows)

        // Same guarantee for the shuffled-view path.
        val shuffledIterator = Iris.load().shuffle(seed = 7L).batchIterator<FP32, Float>(batchSize = 32)
        var shuffledRows = 0
        while (shuffledIterator.hasNext()) {
            shuffledRows += shuffledIterator.next().batchSize
        }
        assertEquals(150, shuffledRows)
    }

    @Test
    fun sameSeedProducesIdenticalSplits() = runBlocking {
        val dataset = Iris.load()

        val (trainA, testA) = dataset.split(splitRatio = 0.8, seed = 42L, stratified = true)
        val (trainB, testB) = dataset.split(splitRatio = 0.8, seed = 42L, stratified = true)

        val labelsA = (0 until trainA.xSize).map { trainA.getY(it) } + (0 until testA.xSize).map { testA.getY(it) }
        val labelsB = (0 until trainB.xSize).map { trainB.getY(it) } + (0 until testB.xSize).map { testB.getY(it) }
        assertEquals(labelsA, labelsB)

        val firstTrainFeaturesA = trainA.getX(0).toList()
        val firstTrainFeaturesB = trainB.getX(0).toList()
        assertEquals(firstTrainFeaturesA, firstTrainFeaturesB)
    }

    @Test
    fun parserRejectsMalformedRowsNamingLineAndColumn() {
        val unknownSpecies = assertFailsWith<IllegalArgumentException> {
            parseIrisCsv("5.1,3.5,1.4,0.2,Iris-unknown")
        }
        assertTrue("line 1" in unknownSpecies.message!!)
        assertTrue("Iris-unknown" in unknownSpecies.message!!)

        val badNumber = assertFailsWith<IllegalArgumentException> {
            parseIrisCsv("abc,3.5,1.4,0.2,Iris-setosa")
        }
        assertTrue("sepalLength" in badNumber.message!!)

        val wrongFieldCount = assertFailsWith<IllegalArgumentException> {
            parseIrisCsv("5.1,3.5,1.4,Iris-setosa")
        }
        assertTrue("expected 5" in wrongFieldCount.message!!)
    }

    private fun classCounts(dataset: Dataset<FloatArray, Int>): Map<Int, Int> =
        (0 until dataset.xSize).groupingBy { idx -> dataset.getY(idx) }.eachCount()

    private fun speciesOfCsvLine(lineIndex: Int): String =
        IRIS_CSV.lines().filter { it.isNotBlank() }[lineIndex].substringAfterLast(",")
}
