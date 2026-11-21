package sk.ainet.io.safetensors

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.deleteIfExists
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path as IoPath
import kotlinx.io.files.SystemFileSystem
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.types.FP32
import kotlin.test.assertContentEquals
import kotlin.test.assertSame

/**
 * JVM-only end-to-end coverage of [StreamingShardedSafeTensorsReader],
 * focused on the `loadTensorViewMapped` entry points, which read a tensor
 * from a mapped shard file so that one above the JVM `ByteArray` limit
 * works (the Gemma 4 PLE token-embedding table).
 *
 * Builds a real tiny single-shard SafeTensors file via [SafeTensorsWriter]
 * and a hand-crafted index JSON, opens via the sharded reader, and
 * verifies that the returned view has the right metadata and reads the
 * file's values.
 */
class StreamingShardedSafeTensorsReaderJvmTest {

    @Test
    fun `loadTensorViewMapped returns a view over the mapped shard`() {
        val tempDir = Files.createTempDirectory("sharded-mapped-")
        try {
            val shardName = "model.safetensors"
            val shardPath = tempDir.resolve(shardName)
            val indexPath = tempDir.resolve("model.safetensors.index.json")

            // Build a 2x2 F32 tensor and write it as a single-shard SafeTensors file.
            val data = floatArrayOf(1f, 2f, 3f, 4f)
            writeSingleTensor(shardPath, name = "test.weight", shape = listOf(2L, 2L), data = data)

            // Hand-craft the index. The sharded reader treats this as a 1-shard
            // sharded model — equivalent to how a single-file gemma-4-e2b-it
            // checkpoint gets routed through the sharded loader.
            val totalSize = Files.size(shardPath)
            val indexJson = """
                {
                  "metadata": {"total_size": $totalSize},
                  "weight_map": {"test.weight": "$shardName"}
                }
            """.trimIndent()
            Files.writeString(indexPath, indexJson)

            runBlocking {
                StreamingShardedSafeTensorsReader.openFromIndex(indexPath.toString()).use { reader ->
                    assertEquals(1, reader.tensors.size)
                    assertEquals("test.weight", reader.tensors[0].name)

                    // 1) Look-up by name overload.
                    val byName = reader.loadTensorViewMapped("test.weight")
                    try {
                        assertEquals(listOf(2, 2), byName.shape.dimensions.toList())
                        assertSame(FP32, byName.format.dtype)
                        assertEquals(MemoryDomain.MMAP_FILE, byName.storage.domain)
                        assertEquals(16L, byName.storage.sizeBytes) // 4 floats × 4 bytes
                        assertContentEquals(data, byName.toFloatArray())
                    } finally {
                        byName.storage.close()
                    }
                    assertEquals(shardPath.toString(), reader.shardPathOf(reader.tensors[0]))

                    // 2) Look-up by ShardedTensorInfo overload — must produce
                    // the same view.
                    val byInfo = reader.loadTensorViewMapped(reader.tensors[0])
                    try {
                        assertEquals(listOf(2, 2), byInfo.shape.dimensions.toList())
                        assertEquals(MemoryDomain.MMAP_FILE, byInfo.storage.domain)
                        assertContentEquals(data, byInfo.toFloatArray())
                    } finally {
                        byInfo.storage.close()
                    }
                }
            }
        } finally {
            tempDir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `loadTensorViewMapped throws on missing tensor name`() {
        val tempDir = Files.createTempDirectory("sharded-mapped-missing-")
        try {
            val shardPath = tempDir.resolve("model.safetensors")
            val indexPath = tempDir.resolve("model.safetensors.index.json")
            writeSingleTensor(shardPath, "real.weight", listOf(1L), floatArrayOf(1f))
            val total = Files.size(shardPath)
            Files.writeString(
                indexPath,
                """{"metadata":{"total_size": $total},"weight_map":{"real.weight":"model.safetensors"}}""",
            )

            runBlocking {
                StreamingShardedSafeTensorsReader.openFromIndex(indexPath.toString()).use { reader ->
                    val thrown = runCatching { reader.loadTensorViewMapped("missing.weight") }
                        .exceptionOrNull()
                    assertNotNull(thrown, "Expected an exception for missing tensor name")
                    assertTrue(thrown is IllegalArgumentException)
                }
            }
        } finally {
            tempDir.toFile().deleteRecursively()
        }
    }

    private fun writeSingleTensor(
        outputPath: Path,
        name: String,
        shape: List<Long>,
        data: FloatArray,
    ) {
        outputPath.deleteIfExists()
        val ioPath = IoPath(outputPath.toString())
        SystemFileSystem.sink(ioPath).buffered().use { sink ->
            SafeTensorsWriter.write(sink) {
                tensorF32(name, shape, data)
            }
        }
        check(outputPath.exists()) { "Failed to write fixture at $outputPath" }
    }
}
