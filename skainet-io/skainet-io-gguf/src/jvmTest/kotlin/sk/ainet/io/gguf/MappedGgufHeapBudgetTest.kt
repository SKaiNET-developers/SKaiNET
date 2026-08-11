package sk.ainet.io.gguf

import sk.ainet.lang.types.FP32
import java.io.File
import java.io.RandomAccessFile
import java.lang.management.ManagementFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The #921 verification gate, host-simulated: a model whose dense weights are
 * **larger than the 512 MB ART large-heap cap** must load and be readable
 * through [MappedGgufWeights] with only O(metadata) managed-heap allocation —
 * the weight bytes live in file-backed mapped pages, exactly as they would on
 * an Android device (`FileChannel.map` is API 1; the identical shared source
 * compiles into the androidMain variant, exercised by
 * `MappedGgufWeightsAndroidHostTest`).
 *
 * Measured with the JVM's per-thread allocation counter — deterministic,
 * unlike sampling heap peaks around GC. No device/emulator is involved; this
 * is the JVM-simulated harness variant of the tracker's verification row.
 *
 * The file is written *sparsely* (header + a few sentinel floats + a
 * `setLength` tail), so the test creates a 640 MB model in milliseconds while
 * the mapped reads still go through real file pages.
 */
class MappedGgufHeapBudgetTest {

    private val threadMx = ManagementFactory.getThreadMXBean() as com.sun.management.ThreadMXBean

    private fun allocatedBytes(): Long = threadMx.getThreadAllocatedBytes(Thread.currentThread().id)

    /** name -> (elements, sentinel flat indices) */
    private val model = listOf(
        Triple("blk0.weight", 64 * 1024 * 1024, intArrayOf(0, 1_000_000, 64 * 1024 * 1024 - 1)),
        Triple("blk1.weight", 64 * 1024 * 1024, intArrayOf(7, 33_554_431)),
        Triple("output.weight", 32 * 1024 * 1024, intArrayOf(12_345, 32 * 1024 * 1024 - 1)),
    )

    private fun sentinelValue(tensor: String, flatIndex: Int): Float =
        (tensor.hashCode() xor flatIndex) * 1e-3f

    /** Sparse GGUF: real header, sentinel floats patched, zero tail via setLength. */
    private fun writeSparseModel(): File {
        val file = File.createTempFile("sparse_640mb_", ".gguf")
        file.deleteOnExit()

        val head = ByteBuffer.allocate(16 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        head.putInt(0x46554747)
        head.putInt(3)
        head.putLong(model.size.toLong())
        head.putLong(1)
        val key = "general.architecture".encodeToByteArray()
        head.putLong(key.size.toLong())
        head.put(key)
        head.putInt(GGUFValueType.STRING.value)
        val value = "test".encodeToByteArray()
        head.putLong(value.size.toLong())
        head.put(value)
        var rel = 0L
        for ((name, elements, _) in model) {
            val nameBytes = name.encodeToByteArray()
            head.putLong(nameBytes.size.toLong())
            head.put(nameBytes)
            head.putInt(1)
            head.putLong(elements.toLong())
            head.putInt(GGMLQuantizationType.F32.value)
            head.putLong(rel)
            rel += elements.toLong() * 4
        }
        val padding = (32 - (head.position() % 32)) % 32
        repeat(padding) { head.put(0) }
        val dataStart = head.position().toLong()

        RandomAccessFile(file, "rw").use { raf ->
            raf.write(head.array(), 0, head.position())
            raf.setLength(dataStart + rel) // sparse zero payload
            // Patch sentinel floats at known flat indices.
            var tensorBase = dataStart
            for ((name, elements, sentinels) in model) {
                for (idx in sentinels) {
                    raf.seek(tensorBase + idx.toLong() * 4)
                    val bits = sentinelValue(name, idx).toRawBits()
                    raf.write(
                        byteArrayOf(
                            (bits and 0xFF).toByte(),
                            ((bits shr 8) and 0xFF).toByte(),
                            ((bits shr 16) and 0xFF).toByte(),
                            ((bits shr 24) and 0xFF).toByte(),
                        )
                    )
                }
                tensorBase += elements.toLong() * 4
            }
        }
        return file
    }

    @Test
    fun `640 MB dense model loads and reads within an O(metadata) heap budget`() {
        val file = writeSparseModel()
        val totalDenseBytes = model.sumOf { it.second.toLong() * 4 }
        assertTrue(totalDenseBytes > 512L * 1024 * 1024, "model must exceed the 512 MB ART budget")
        try {
            // Warm-up on a tiny file: classloading and JIT.
            MappedGgufWeightsTest.writeGguf(
                listOf(
                    MappedGgufWeightsTest.Companion.GgufTestTensor(
                        "w", GGMLQuantizationType.F32, 8,
                        MappedGgufWeightsTest.f32ToBytes(FloatArray(8) { it.toFloat() }),
                    )
                )
            ).let { warm ->
                MappedGgufWeights.open(warm.absolutePath).use { it.mappedFloatTensor<FP32>("w")[3] }
                warm.delete()
            }

            val before = allocatedBytes()

            var checksum = 0.0
            MappedGgufWeights.open(file.absolutePath).use { weights ->
                for ((name, elements, sentinels) in model) {
                    val tensor = weights.mappedFloatTensor<FP32>(name)
                    assertEquals(elements, tensor.shape.volume, name)
                    // Sample-read across the whole tensor (touches mapped pages,
                    // allocates nothing on the heap) …
                    var i = 0
                    while (i < elements) {
                        checksum += tensor[i]
                        i += 1_048_576
                    }
                    // … and verify the sentinels round-trip through the mapping.
                    for (idx in sentinels) {
                        assertEquals(sentinelValue(name, idx), tensor[idx], "$name[$idx]")
                    }
                }
            }

            val allocated = allocatedBytes() - before
            println(
                "mapped GGUF load: dense=${totalDenseBytes / (1024 * 1024)} MB, " +
                    "heap allocated=${allocated / 1024} KB " +
                    "(${"%.4f".format(allocated / totalDenseBytes.toDouble())}x of dense size), checksum=$checksum",
            )

            // O(metadata): parsing + view objects. 8 MB is ~1.2% of the dense
            // size and orders of magnitude under the 512 MB ART budget; a heap
            // materialization of even one tensor (256 MB) trips this instantly.
            val budget = 8L * 1024 * 1024
            assertTrue(
                allocated in 0..budget,
                "loading a ${totalDenseBytes / (1024 * 1024)} MB dense model allocated " +
                    "${allocated / (1024 * 1024)} MB on the managed heap — weight bytes are " +
                    "no longer staying in mapped pages (#921)",
            )
        } finally {
            file.delete()
        }
    }
}
