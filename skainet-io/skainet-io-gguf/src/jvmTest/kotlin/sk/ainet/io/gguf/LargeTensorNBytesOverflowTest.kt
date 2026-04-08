package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Regression test for nBytes Int overflow with tensors > 2 GB.
 *
 * Bug: StreamingTensorInfo.nBytes was Int, which overflowed for tensors
 * whose byte size exceeds Int.MAX_VALUE (~2 GB). This blocked loading
 * Gemma 4 E4B (per_layer_token_embd.weight is ~1.4 GB in Q4_K_M but
 * the Long→Int cast overflowed earlier in the arithmetic chain for
 * larger tensors).
 *
 * Root cause: `(numBlocks * typeSize).toInt()` silently truncated a
 * Long result that exceeded Int.MAX_VALUE.
 *
 * Fix: Changed StreamingTensorInfo.nBytes from Int to Long, and kept
 * all intermediate arithmetic in Long.
 */
class LargeTensorNBytesOverflowTest {

    /**
     * Verify that nBytes computation stays in Long and does not overflow
     * for a tensor with > 2 GB byte size.
     *
     * Simulates a tensor like Gemma 4 E4B's per_layer_token_embd.weight:
     * shape [262144, 10752], Q4_K format.
     *
     * Q4_K: blockSize=256, typeSize=144
     * nElements = 262144 * 10752 = 2,818,572,288
     * numBlocks = 2,818,572,288 / 256 = 11,009,892
     * nBytes = 11,009,892 * 144 = 1,585,424,448 (fits in Int)
     *
     * But for a larger tensor (e.g., shape [524288, 10752]):
     * nElements = 524288 * 10752 = 5,637,144,576
     * numBlocks = 5,637,144,576 / 256 = 22,019,784
     * nBytes = 22,019,784 * 144 = 3,170,848,896 (> Int.MAX_VALUE!)
     */
    @Test
    fun nBytesComputationDoesNotOverflowForLargeTensors() {
        // Create a minimal GGUF with a large tensor that would overflow Int
        val largeTensorGguf = buildLargeTensorGguf(
            tensorName = "large_weight",
            dims = listOf(524288UL, 10752UL), // > 2GB in Q4_K
            ggmlType = GGMLQuantizationType.Q4_K
        )

        val source = bytesAsSource(largeTensorGguf)
        val reader = StreamingGGUFReader.open(source)

        assertEquals(1, reader.tensors.size)
        val tensor = reader.tensors[0]

        // The key assertion: nBytes must be positive and correct
        val expectedElements = 524288L * 10752L
        val expectedBlocks = expectedElements / 256
        val expectedBytes = expectedBlocks * 144

        assertEquals("large_weight", tensor.name)
        assertEquals(expectedElements, tensor.nElements)
        assertTrue(tensor.nBytes > 0, "nBytes must be positive, got ${tensor.nBytes}")
        assertEquals(expectedBytes, tensor.nBytes,
            "nBytes computation should not overflow: expected $expectedBytes, got ${tensor.nBytes}")
        assertTrue(tensor.nBytes > Int.MAX_VALUE,
            "This tensor should exceed Int.MAX_VALUE to test the fix")
    }

    @Test
    fun nBytesCorrectForNormalSizedTensor() {
        // Normal tensor that fits in Int — should still work
        val normalGguf = buildLargeTensorGguf(
            tensorName = "normal_weight",
            dims = listOf(256UL, 256UL), // 65536 elements, Q4_K
            ggmlType = GGMLQuantizationType.Q4_K
        )

        val source = bytesAsSource(normalGguf)
        val reader = StreamingGGUFReader.open(source)

        val tensor = reader.tensors[0]
        val expectedElements = 256L * 256L
        val expectedBlocks = expectedElements / 256
        val expectedBytes = expectedBlocks * 144

        assertEquals(expectedElements, tensor.nElements)
        assertEquals(expectedBytes, tensor.nBytes)
    }

    @Test
    fun loadTensorDataRejectsLargeTensor() {
        val largeTensorGguf = buildLargeTensorGguf(
            tensorName = "huge_weight",
            dims = listOf(524288UL, 10752UL),
            ggmlType = GGMLQuantizationType.Q4_K
        )

        val source = bytesAsSource(largeTensorGguf)
        val reader = StreamingGGUFReader.open(source)
        val tensor = reader.tensors[0]

        // loadTensorData should reject > 2GB tensors with a clear error
        assertFailsWith<IllegalArgumentException> {
            reader.loadTensorData(tensor)
        }
    }

    @Test
    fun loadTensorStorageMappedWorksForLargeTensor() {
        val largeTensorGguf = buildLargeTensorGguf(
            tensorName = "huge_weight",
            dims = listOf(524288UL, 10752UL),
            ggmlType = GGMLQuantizationType.Q4_K
        )

        val source = bytesAsSource(largeTensorGguf)
        val reader = StreamingGGUFReader.open(source)
        val tensor = reader.tensors[0]

        // loadTensorStorageMapped should work — it creates a FileBacked handle
        // without loading data into heap
        val storage = reader.loadTensorStorageMapped(tensor, "/fake/path.gguf")
        assertTrue(storage.isFileBacked)
        assertEquals(tensor.nBytes, storage.buffer.sizeInBytes)
    }

    @Test
    fun gemma4StyleTensorDoesNotOverflow() {
        // Simulate the exact Gemma 4 E4B tensor that triggered the bug:
        // per_layer_token_embd.weight: shape [262144, 10752], Q4_K_M
        // This one fits in Int but was failing due to intermediate overflow
        val gguf = buildLargeTensorGguf(
            tensorName = "per_layer_token_embd.weight",
            dims = listOf(262144UL, 10752UL),
            ggmlType = GGMLQuantizationType.Q4_K
        )

        val source = bytesAsSource(gguf)
        val reader = StreamingGGUFReader.open(source)
        val tensor = reader.tensors[0]

        val expectedElements = 262144L * 10752L
        val expectedBlocks = expectedElements / 256
        val expectedBytes = expectedBlocks * 144 // 1,585,424,448 bytes

        assertEquals(expectedElements, tensor.nElements)
        assertEquals(expectedBytes, tensor.nBytes)
        assertTrue(tensor.nBytes > 0, "nBytes must not be negative (overflow)")
        assertTrue(tensor.nBytes < Int.MAX_VALUE,
            "Gemma 4 E4B PLE weight should fit in Int: ${tensor.nBytes}")
    }

    // ========== Helpers ==========

    /**
     * Build a minimal valid GGUF v3 file with one tensor (metadata only, no actual data).
     *
     * GGUF format:
     * - Magic (4 bytes): 0x46554747
     * - Version (4 bytes): 3
     * - Tensor count (8 bytes)
     * - KV count (8 bytes): 0
     * - Tensor info entries
     * - Data section (empty — we only test metadata parsing)
     */
    private fun buildLargeTensorGguf(
        tensorName: String,
        dims: List<ULong>,
        ggmlType: GGMLQuantizationType
    ): ByteArray {
        val nameBytes = tensorName.encodeToByteArray()
        // Calculate size:
        // Header: 4 + 4 + 8 + 8 = 24
        // Tensor info: 8 (name len) + name + 4 (ndims) + 8*ndims (dims) + 4 (type) + 8 (offset)
        val tensorInfoSize = 8 + nameBytes.size + 4 + 8 * dims.size + 4 + 8
        val totalSize = 24 + tensorInfoSize + 32 // +32 for alignment padding

        val buf = ByteArray(totalSize)
        var pos = 0

        // Magic
        writeUInt(buf, pos, GGUF_MAGIC); pos += 4
        // Version
        writeUInt(buf, pos, 3u); pos += 4
        // Tensor count
        writeULong(buf, pos, 1u); pos += 8
        // KV count
        writeULong(buf, pos, 0u); pos += 8

        // Tensor info: name length
        writeULong(buf, pos, nameBytes.size.toULong()); pos += 8
        // Tensor info: name
        nameBytes.copyInto(buf, pos); pos += nameBytes.size
        // Tensor info: ndims
        writeUInt(buf, pos, dims.size.toUInt()); pos += 4
        // Tensor info: dims
        for (d in dims) {
            writeULong(buf, pos, d); pos += 8
        }
        // Tensor info: type
        writeUInt(buf, pos, ggmlType.value.toUInt()); pos += 4
        // Tensor info: relative offset
        writeULong(buf, pos, 0u); pos += 8

        return buf
    }

    private fun writeUInt(buf: ByteArray, pos: Int, value: UInt) {
        val v = value.toInt()
        buf[pos] = (v and 0xFF).toByte()
        buf[pos + 1] = ((v shr 8) and 0xFF).toByte()
        buf[pos + 2] = ((v shr 16) and 0xFF).toByte()
        buf[pos + 3] = ((v shr 24) and 0xFF).toByte()
    }

    private fun writeULong(buf: ByteArray, pos: Int, value: ULong) {
        val v = value.toLong()
        for (i in 0 until 8) {
            buf[pos + i] = ((v shr (i * 8)) and 0xFF).toByte()
        }
    }

    private fun bytesAsSource(bytes: ByteArray): RandomAccessSource {
        return object : RandomAccessSource {
            override val size: Long get() = bytes.size.toLong()
            override fun readAt(position: Long, length: Int): ByteArray =
                bytes.copyOfRange(position.toInt(), position.toInt() + length)
            override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
                bytes.copyInto(buffer, offset, position.toInt(), position.toInt() + length)
                return length
            }
            override fun close() {}
        }
    }
}
