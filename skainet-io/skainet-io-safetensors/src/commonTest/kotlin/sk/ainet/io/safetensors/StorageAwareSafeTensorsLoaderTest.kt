@file:Suppress("DEPRECATION") // LogicalDType legacy path kept under test until removal (SKEEP-003 #1014)

package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.storage.LogicalDType
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.Ownership
import sk.ainet.lang.tensor.storage.Residency
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests for [StorageAwareSafeTensorsLoader].
 */
class StorageAwareSafeTensorsLoaderTest {

    /**
     * Create a minimal valid SafeTensors file in memory.
     *
     * SafeTensors format: 8-byte header size (LE) + JSON header + tensor data.
     */
    private fun createSafeTensorsBytes(
        tensors: Map<String, Pair<String, List<Long>>> = mapOf(
            "weight" to ("F32" to listOf(2L, 3L))
        )
    ): ByteArray {
        // Build tensor data and header entries
        val tensorEntries = mutableListOf<String>()
        val dataChunks = mutableListOf<ByteArray>()
        var offset = 0L

        for ((name, info) in tensors) {
            val (dtype, shape) = info
            val bytesPerElement = when (dtype) {
                "F32" -> 4
                "F16" -> 2
                "I32" -> 4
                "I8" -> 1
                else -> 4
            }
            val elementCount = if (shape.isEmpty()) 1L else shape.fold(1L) { a, b -> a * b }
            val sizeInBytes = elementCount * bytesPerElement
            val data = ByteArray(sizeInBytes.toInt())
            // Fill with recognizable pattern
            for (i in data.indices) data[i] = (i % 256).toByte()
            dataChunks.add(data)

            val shapeStr = shape.joinToString(",")
            tensorEntries.add(
                "\"$name\":{\"dtype\":\"$dtype\",\"shape\":[$shapeStr],\"data_offsets\":[$offset,${offset + sizeInBytes}]}"
            )
            offset += sizeInBytes
        }

        val headerJson = "{${tensorEntries.joinToString(",")}}"
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        // 8 bytes header size (LE) + header + data
        val result = ByteArray(8 + headerBytes.size + dataChunks.sumOf { it.size })
        // Write header size as LE u64
        for (i in 0 until 8) {
            result[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(result, 8)
        var dataOffset = 8 + headerBytes.size
        for (chunk in dataChunks) {
            chunk.copyInto(result, dataOffset)
            dataOffset += chunk.size
        }
        return result
    }

    private fun bytesAsSource(bytes: ByteArray): RandomAccessSource {
        return object : RandomAccessSource {
            override val size: Long get() = bytes.size.toLong()

            override fun readAt(offset: Long, length: Int): ByteArray {
                return bytes.copyOfRange(offset.toInt(), offset.toInt() + length)
            }

            override fun readAt(offset: Long, buffer: ByteArray, bufferOffset: Int, length: Int): Int {
                bytes.copyInto(buffer, bufferOffset, offset.toInt(), offset.toInt() + length)
                return length
            }

            override fun close() {}
        }
    }

    // --- Heap-loaded (borrowed) mode ---

    @Test
    fun loadAllBorrowed_returnsCorrectStorage() {
        val fileBytes = createSafeTensorsBytes()
        val loader = StorageAwareSafeTensorsLoader(
            sourceProvider = { bytesAsSource(fileBytes) }
        )

        val tensors = loader.loadAll()
        assertEquals(1, tensors.size)
        assertTrue(tensors.containsKey("weight"))

        val storage = tensors["weight"]!!
        assertEquals(LogicalDType.FLOAT32, storage.logicalType)
        assertEquals(Ownership.BORROWED, storage.ownership)
        assertFalse(storage.isFileBacked)
        assertEquals(6L, storage.elementCount) // 2 * 3
    }

    // --- File-backed (zero-copy) mode ---

    @Test
    fun loadAllMapped_returnsFileBackedStorage() {
        val fileBytes = createSafeTensorsBytes()
        val loader = StorageAwareSafeTensorsLoader(
            sourceProvider = { bytesAsSource(fileBytes) },
            filePath = "/test/model.safetensors"
        )

        val tensors = loader.loadAll()
        val storage = tensors["weight"]!!
        assertTrue(storage.isFileBacked)
        assertEquals(Ownership.FILE_BACKED, storage.ownership)
        assertEquals(MemoryDomain.MMAP_FILE, storage.placement.domain)
        assertEquals(Residency.PERSISTENT, storage.placement.residency)
        assertFalse(storage.isMutable)
    }

    // --- Single tensor load ---

    @Test
    fun loadSingleTensor() {
        val fileBytes = createSafeTensorsBytes(
            mapOf(
                "a" to ("F32" to listOf(4L)),
                "b" to ("F32" to listOf(8L))
            )
        )
        val loader = StorageAwareSafeTensorsLoader(
            sourceProvider = { bytesAsSource(fileBytes) }
        )

        val storageA = loader.load("a")
        assertEquals(4L, storageA.elementCount)

        val storageB = loader.load("b")
        assertEquals(8L, storageB.elementCount)
    }

    @Test
    fun loadMissingTensorThrows() {
        val fileBytes = createSafeTensorsBytes()
        val loader = StorageAwareSafeTensorsLoader(
            sourceProvider = { bytesAsSource(fileBytes) }
        )

        assertFailsWith<IllegalArgumentException> {
            loader.load("nonexistent")
        }
    }

    // --- List tensors ---

    @Test
    fun listTensorsReturnsMetadata() {
        val fileBytes = createSafeTensorsBytes(
            mapOf(
                "embed" to ("F32" to listOf(100L, 64L)),
                "bias" to ("F32" to listOf(64L))
            )
        )
        val loader = StorageAwareSafeTensorsLoader(
            sourceProvider = { bytesAsSource(fileBytes) }
        )

        val infos = loader.listTensors()
        assertEquals(2, infos.size)
        assertEquals(setOf("embed", "bias"), infos.map { it.name }.toSet())
    }

    // --- Progress callback ---

    @Test
    fun progressCallbackIsCalled() {
        val fileBytes = createSafeTensorsBytes(
            mapOf(
                "a" to ("F32" to listOf(4L)),
                "b" to ("F32" to listOf(8L))
            )
        )
        val progressCalls = mutableListOf<Triple<Long, Long, String?>>()
        val loader = StorageAwareSafeTensorsLoader(
            sourceProvider = { bytesAsSource(fileBytes) },
            onProgress = { current, total, name -> progressCalls.add(Triple(current, total, name)) }
        )

        loader.loadAll()
        assertEquals(2, progressCalls.size)
        assertEquals(2L, progressCalls[1].second) // total
    }
}
