package sk.ainet.io.safetensors

import kotlinx.coroutines.test.runTest
import sk.ainet.io.ByteArrayRandomAccessSource
import sk.ainet.io.RandomAccessSource
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * [StreamingShardedSafeTensorsReader.openFromIndexContent]: a sharded checkpoint whose shards the
 * caller provides — the entry point for a browser or Wasm host, where there is no path to resolve
 * (JVM ↔ wasm IO parity). Runs on every target; the sources are in-memory.
 */
class ShardedSafeTensorsReaderSourceTest {

    /** A source that records whether it was closed and may claim to be a file. */
    private class Probe(bytes: ByteArray, override val filePath: String? = null) : RandomAccessSource {
        private val inner = ByteArrayRandomAccessSource(bytes)
        var closed = 0
        override val size: Long get() = inner.size
        override fun readAt(position: Long, length: Int): ByteArray = inner.readAt(position, length)
        override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int = inner.readAt(position, buffer, offset, length)
        override fun close() { closed++ }
    }

    private val shards = ShardFixtures.twoShards()
    private val index = ShardFixtures.twoShardIndex(shards)

    @Test
    fun theProviderIsAskedOncePerShardAndTensorsArriveNameSorted() = runTest {
        val asked = ArrayList<String>()
        val opened = HashMap<String, Probe>()
        val reader = StreamingShardedSafeTensorsReader.openFromIndexContent(index, { name ->
            asked += name
            shards[name]?.let { Probe(it).also { p -> opened[name] = p } }
        })

        assertEquals(listOf(ShardFixtures.SHARD_1, ShardFixtures.SHARD_2), asked, "each shard is opened exactly once")
        assertEquals(listOf("alpha.weight", "bravo.weight", "zulu.weight"), reader.tensors.map { it.name })
        assertTrue(reader.isComplete)
        assertEquals(listOf(ShardFixtures.SHARD_1, ShardFixtures.SHARD_2), reader.loadedShards)
        assertEquals(2, reader.index.shardCount)
        assertEquals(shards.values.sumOf { it.size.toLong() }, reader.totalSize)
        assertEquals("pt", reader.metadata["format"], "shard metadata is merged")

        val zulu = reader.tensors.single { it.name == "zulu.weight" }
        assertEquals("2/2", zulu.shardLocation)
        assertContentEquals(ShardFixtures.fp32Bytes(ShardFixtures.zuluValues), reader.loadTensorData(zulu))
        assertContentEquals(ShardFixtures.fp32Bytes(ShardFixtures.alphaValues), reader.loadTensorData("alpha.weight"))
        assertContentEquals(ShardFixtures.bf16Bytes(ShardFixtures.bravoValues), reader.loadTensorData("bravo.weight"))

        val mapped = assertFailsWith<UnsupportedOperationException> { reader.loadTensorViewMapped(zulu) }
        assertTrue(mapped.message!!.contains("filePath"), mapped.message)

        assertTrue(opened.values.all { it.closed == 0 }, "sources stay open while the reader is")
        reader.close()
        assertTrue(opened.values.all { it.closed == 1 }, "close() closes every shard source once: ${opened.mapValues { it.value.closed }}")
    }

    @Test
    fun aSourceThatNamesItsFileKeepsMappedLoadingAvailable() = runTest {
        val reader = StreamingShardedSafeTensorsReader.openFromIndexContent(index, { name ->
            shards[name]?.let { Probe(it, filePath = "/models/$name") }
        })
        // The reader knows the file behind the shard, which is what mapped loading needs.
        // (Mapping itself needs a real file and a target with mmap: see the JVM test.)
        val alpha = reader.tensors.single { it.name == "alpha.weight" }
        assertEquals("/models/${ShardFixtures.SHARD_1}", reader.shardPathOf(alpha))
        assertEquals(ShardFixtures.alphaValues.size * 4L, alpha.sizeInBytes)
        reader.close()
    }

    @Test
    fun aMissingShardIsIncompleteUnlessPartialLoadsAreAllowed() = runTest {
        val opened = ArrayList<Probe>()
        val onlyFirst: suspend (String) -> RandomAccessSource? = { name ->
            if (name == ShardFixtures.SHARD_1) Probe(shards.getValue(name)).also { opened += it } else null
        }

        val error = assertFailsWith<SafeTensorsShardException.IncompleteShard> {
            StreamingShardedSafeTensorsReader.openFromIndexContent(index, onlyFirst)
        }
        assertEquals(listOf(ShardFixtures.SHARD_2), error.missingShards)
        assertEquals(1, error.loadedCount)
        assertTrue(opened.all { it.closed == 1 }, "the shard that was opened is closed again before the failure propagates")

        val partial = StreamingShardedSafeTensorsReader.openFromIndexContent(index, onlyFirst, allowPartial = true)
        assertFalse(partial.isComplete)
        assertEquals(listOf(ShardFixtures.SHARD_2), partial.missingShards)
        assertEquals(listOf("alpha.weight", "bravo.weight"), partial.tensors.map { it.name })
        assertFailsWith<IllegalArgumentException> { partial.loadTensorData("zulu.weight") }
        partial.close()
    }

    @Test
    fun aShardThatFailsToParseClosesEverythingUnlessPartialLoadsAreAllowed() = runTest {
        val opened = HashMap<String, Probe>()
        val corruptSecond: suspend (String) -> RandomAccessSource? = { name ->
            val bytes = if (name == ShardFixtures.SHARD_2) ByteArray(64) { 0x7F } else shards.getValue(name)
            Probe(bytes).also { opened[name] = it }
        }

        assertFailsWith<IllegalArgumentException> { StreamingShardedSafeTensorsReader.openFromIndexContent(index, corruptSecond) }
        assertEquals(2, opened.size)
        assertTrue(opened.values.all { it.closed >= 1 }, "every opened source is closed on failure: ${opened.mapValues { it.value.closed }}")

        opened.clear()
        val partial = StreamingShardedSafeTensorsReader.openFromIndexContent(index, corruptSecond, allowPartial = true)
        assertEquals(listOf(ShardFixtures.SHARD_2), partial.missingShards)
        assertEquals(listOf("alpha.weight", "bravo.weight"), partial.tensors.map { it.name })
        assertEquals(1, opened.getValue(ShardFixtures.SHARD_2).closed, "the corrupt shard's source is released")
        partial.close()
    }

    @Test
    fun aProviderThatThrowsPropagatesUnlessPartialLoadsAreAllowed() = runTest {
        val throwing: suspend (String) -> RandomAccessSource? = { name ->
            if (name == ShardFixtures.SHARD_2) throw IllegalStateException("network down") else Probe(shards.getValue(name))
        }
        val error = assertFailsWith<IllegalStateException> { StreamingShardedSafeTensorsReader.openFromIndexContent(index, throwing) }
        assertEquals("network down", error.message)

        val partial = StreamingShardedSafeTensorsReader.openFromIndexContent(index, throwing, allowPartial = true)
        assertEquals(listOf(ShardFixtures.SHARD_2), partial.missingShards)
        partial.close()
    }

    @Test
    fun theParsedIndexEntryPointTakesAnIndexObject() = runTest {
        val parsed = SafeTensorsIndexParser.parse(index)
        val reader = StreamingShardedSafeTensorsReader.openFromParsedIndex(parsed, { shards[it]?.let(::ByteArrayRandomAccessSource) })
        assertEquals(3, reader.tensors.size)
        assertEquals(parsed, reader.index)
        reader.close()
    }
}
