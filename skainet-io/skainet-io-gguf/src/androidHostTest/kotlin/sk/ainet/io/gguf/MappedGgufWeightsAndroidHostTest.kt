package sk.ainet.io.gguf

import sk.ainet.io.JvmFileBackedResolver
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.types.FP32
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Host-side test of the *Android compilation* of the mmap weight path (#921):
 * this source set compiles against androidMain (android.jar nio API), so it
 * proves `MappedGgufWeights`, `MmapFloatTensorData` and the shared
 * `JvmFileBackedResolver` all build and behave on the Android variant —
 * without a device or emulator (the precise allocation-counter budget gate
 * lives in jvmTest's `MappedGgufHeapBudgetTest`, which exercises the
 * byte-identical shared source).
 *
 * Includes a coarse heap check on a sparse 96 MB model: after load + reads,
 * used-heap growth stays far under the payload size (lenient bound — host GC
 * is not deterministic; the strict gate is the jvmTest allocation counter).
 */
class MappedGgufWeightsAndroidHostTest {

    private fun writeSparseF32Gguf(file: File, tensorName: String, elements: Int, sentinels: Map<Int, Float>): File {
        val head = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)
        head.putInt(0x46554747)
        head.putInt(3)
        head.putLong(1)
        head.putLong(1)
        val key = "general.architecture".encodeToByteArray()
        head.putLong(key.size.toLong())
        head.put(key)
        head.putInt(GGUFValueType.STRING.value)
        val value = "test".encodeToByteArray()
        head.putLong(value.size.toLong())
        head.put(value)
        val nameBytes = tensorName.encodeToByteArray()
        head.putLong(nameBytes.size.toLong())
        head.put(nameBytes)
        head.putInt(1)
        head.putLong(elements.toLong())
        head.putInt(GGMLQuantizationType.F32.value)
        head.putLong(0)
        val padding = (32 - (head.position() % 32)) % 32
        repeat(padding) { head.put(0) }
        val dataStart = head.position().toLong()

        RandomAccessFile(file, "rw").use { raf ->
            raf.write(head.array(), 0, head.position())
            raf.setLength(dataStart + elements.toLong() * 4)
            for ((idx, v) in sentinels) {
                raf.seek(dataStart + idx.toLong() * 4)
                val bits = v.toRawBits()
                raf.write(
                    byteArrayOf(
                        (bits and 0xFF).toByte(),
                        ((bits shr 8) and 0xFF).toByte(),
                        ((bits shr 16) and 0xFF).toByte(),
                        ((bits shr 24) and 0xFF).toByte(),
                    )
                )
            }
        }
        return file
    }

    @Test
    fun `android-compiled mapped weight path loads a 96 MB tensor without heap-sized allocation`() {
        val elements = 24 * 1024 * 1024 // 96 MB dense F32
        val sentinels = mapOf(0 to 1.5f, 12_345_678 to -2.25f, elements - 1 to 3.125f)
        val file = File.createTempFile("android_mmap_", ".gguf")
        file.deleteOnExit()
        writeSparseF32Gguf(file, "big.weight", elements, sentinels)
        try {
            val rt = Runtime.getRuntime()
            System.gc()
            val usedBefore = rt.totalMemory() - rt.freeMemory()

            var checksum = 0.0
            MappedGgufWeights.open(file.absolutePath).use { weights ->
                val tensor = weights.mappedFloatTensor<FP32>("big.weight")
                assertEquals(elements, tensor.shape.volume)
                var i = 0
                while (i < elements) {
                    checksum += tensor[i]
                    i += 262_144
                }
                for ((idx, v) in sentinels) {
                    assertEquals(v, tensor[idx], "big.weight[$idx]")
                }

                // FileBacked descriptor + the shared resolver, on the android variant.
                val storage = weights.mappedStorage("big.weight")
                assertTrue(storage.isFileBacked)
                val accessor = JvmFileBackedResolver.resolveFileBacked(storage.buffer as BufferHandle.FileBacked)
                try {
                    // First 4 sentinel bytes come back identical through the mmap accessor.
                    val viaAccessor = accessor.readBytes(0, 4)
                    val expectedBits = 1.5f.toRawBits()
                    assertContentEquals(
                        byteArrayOf(
                            (expectedBits and 0xFF).toByte(),
                            ((expectedBits shr 8) and 0xFF).toByte(),
                            ((expectedBits shr 16) and 0xFF).toByte(),
                            ((expectedBits shr 24) and 0xFF).toByte(),
                        ),
                        viaAccessor,
                    )
                } finally {
                    accessor.close()
                }
            }

            System.gc()
            val usedAfter = rt.totalMemory() - rt.freeMemory()
            val growth = usedAfter - usedBefore
            println("android host mapped load: payload=96 MB, used-heap growth=${growth / 1024} KB, checksum=$checksum")
            // Lenient (GC nondeterminism): far below the 96 MB payload — a heap
            // materialization would add >= 96 MB here.
            assertTrue(
                growth < 48L * 1024 * 1024,
                "used heap grew by ${growth / (1024 * 1024)} MB for a 96 MB mapped payload — " +
                    "weights are being materialized on the managed heap (#921)",
            )
        } finally {
            file.delete()
        }
    }
}
