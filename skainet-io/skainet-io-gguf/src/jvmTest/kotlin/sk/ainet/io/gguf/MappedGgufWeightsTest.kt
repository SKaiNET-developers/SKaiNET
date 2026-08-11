package sk.ainet.io.gguf

import sk.ainet.io.JvmFileBackedResolver
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.Placement
import sk.ainet.lang.types.FP32
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Functional coverage for [MappedGgufWeights] (#921): mapped F32 views,
 * FileBacked descriptors resolved through the (JVM+Android shared)
 * [JvmFileBackedResolver], packed-byte fallback, and the fail-fast guards.
 */
class MappedGgufWeightsTest {

    @Test
    fun `mapped F32 view, FileBacked storage and packed bytes agree with the file`() {
        val f32Values = FloatArray(1024) { (it - 512) * 0.25f }
        val q80Payload = ByteArray(34 * 4) { (it * 7 + 3).toByte() } // 4 blocks of Q8_0
        val file = writeGguf(
            listOf(
                GgufTestTensor("dense.f32", GGMLQuantizationType.F32, 1024, f32ToBytes(f32Values)),
                GgufTestTensor("packed.q80", GGMLQuantizationType.Q8_0, 128, q80Payload),
            )
        )
        try {
            MappedGgufWeights.open(file.absolutePath).use { weights ->
                assertEquals(2, weights.tensors.size)

                // Zero-heap mapped view: values read straight from mapped pages.
                val dense = weights.mappedFloatTensor<FP32>("dense.f32")
                assertEquals(1024, dense.shape.volume)
                for (i in intArrayOf(0, 1, 511, 512, 1023)) {
                    assertEquals(f32Values[i], dense[i], "flat index $i")
                }

                // FileBacked descriptor + shared resolver: bytes match the payload.
                val storage = weights.mappedStorage("packed.q80")
                assertTrue(storage.isFileBacked, "storage should be FileBacked")
                assertEquals(Placement.MMAP_WEIGHTS, storage.placement)
                val handle = storage.buffer as BufferHandle.FileBacked
                val accessor = JvmFileBackedResolver.resolveFileBacked(handle)
                try {
                    assertContentEquals(q80Payload, accessor.readBytes(0, q80Payload.size))
                } finally {
                    accessor.close()
                }

                // copyMaterialize with the resolver turns FileBacked into Owned bytes.
                val materialized = storage.copyMaterialize(JvmFileBackedResolver.createResolver())
                val owned = materialized.buffer as BufferHandle.Owned
                assertContentEquals(q80Payload, owned.data)

                // Heap fallback for packed kernels.
                assertContentEquals(q80Payload, weights.packedBytes("packed.q80"))

                // Guards.
                assertFailsWith<IllegalArgumentException> { weights.mappedFloatTensor<FP32>("packed.q80") }
                assertFailsWith<IllegalArgumentException> { weights.info("missing") }
            }
        } finally {
            file.delete()
        }
    }

    companion object {
        fun f32ToBytes(values: FloatArray): ByteArray {
            val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            values.forEach { buf.putFloat(it) }
            return buf.array()
        }

        data class GgufTestTensor(
            val name: String,
            val type: GGMLQuantizationType,
            val elementCount: Long,
            val data: ByteArray,
        )

        /** Write a minimal GGUF v3 file (32-byte aligned data section). */
        fun writeGguf(tensors: List<GgufTestTensor>, file: File = File.createTempFile("mapped_gguf_", ".gguf")): File {
            file.deleteOnExit()
            val head = ByteBuffer.allocate(16 * 1024).order(ByteOrder.LITTLE_ENDIAN)
            head.putInt(0x46554747) // "GGUF"
            head.putInt(3)
            head.putLong(tensors.size.toLong())
            head.putLong(1)
            val key = "general.architecture".encodeToByteArray()
            head.putLong(key.size.toLong())
            head.put(key)
            head.putInt(GGUFValueType.STRING.value)
            val value = "test".encodeToByteArray()
            head.putLong(value.size.toLong())
            head.put(value)
            var dataOffset = 0L
            for (t in tensors) {
                val name = t.name.encodeToByteArray()
                head.putLong(name.size.toLong())
                head.put(name)
                head.putInt(1)
                head.putLong(t.elementCount)
                head.putInt(t.type.value)
                head.putLong(dataOffset)
                dataOffset += padded(t.data.size)
            }
            val padding = (32 - (head.position() % 32)) % 32
            repeat(padding) { head.put(0) }
            RandomAccessFile(file, "rw").use { raf ->
                raf.write(head.array(), 0, head.position())
                for (t in tensors) {
                    raf.write(t.data)
                    repeat(padded(t.data.size) - t.data.size) { raf.write(0) }
                }
            }
            return file
        }

        private fun padded(size: Int): Int = ((size + 31) / 32) * 32
    }
}
