package sk.ainet.io.safetensors

import kotlinx.coroutines.runBlocking
import org.junit.Test
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1256: `tensorFilter` on the single-file loader — parity with the sharded
 * loader. A checkpoint that also carries tensors the requested dtype cannot
 * accept (here an INT64 index table next to FP32 weights) must be loadable
 * selectively; without the filter the per-arm `require` still fails as before.
 */
class SafeTensorsParametersLoaderFilterTest {

    /** One F32 `weight` [2,2] and one I64 `positions` [3] in a single file. */
    private fun mixedFile(): File {
        val weight = floatArrayOf(1f, 2f, 3f, 4f)
        val positions = longArrayOf(0L, 1L, 2L)
        val weightBytes = ByteBuffer.allocate(weight.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            .apply { weight.forEach { putFloat(it) } }.array()
        val posBytes = ByteBuffer.allocate(positions.size * 8).order(ByteOrder.LITTLE_ENDIAN)
            .apply { positions.forEach { putLong(it) } }.array()
        val header = """{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,${weightBytes.size}]},""" +
            """"positions":{"dtype":"I64","shape":[3],"data_offsets":[${weightBytes.size},${weightBytes.size + posBytes.size}]}}"""
        val headerBytes = header.toByteArray(Charsets.UTF_8)
        val file = Files.createTempFile("filter_test", ".safetensors").toFile().also { it.deleteOnExit() }
        file.outputStream().use { os ->
            os.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.size.toLong()).array())
            os.write(headerBytes)
            os.write(weightBytes)
            os.write(posBytes)
        }
        return file
    }

    @Test
    fun filter_skips_unwanted_tensors_and_progress_counts_only_the_kept_ones() = runBlocking {
        val file = mixedFile()
        val delivered = linkedMapOf<String, Tensor<FP32, Float>>()
        val totals = mutableSetOf<Long>()
        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) },
            onProgress = { _, total, _ -> totals += total },
            tensorFilter = { it.name == "weight" },
        )
        loader.load<FP32, Float>(DirectCpuExecutionContext(), FP32::class) { name, t -> delivered[name] = t }

        assertEquals(setOf("weight"), delivered.keys)
        assertEquals(setOf(1L), totals, "progress total must reflect the filtered count")
        val data = delivered.getValue("weight").data as FloatArrayTensorData<*>
        assertTrue(floatArrayOf(1f, 2f, 3f, 4f).contentEquals(data.buffer))
    }

    @Test
    fun without_filter_the_unaccepted_dtype_still_fails_as_before() = runBlocking {
        val file = mixedFile()
        val loader = SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(file) })
        assertFailsWith<IllegalArgumentException> {
            loader.load<FP32, Float>(DirectCpuExecutionContext(), FP32::class) { _, _ -> }
        }
        Unit
    }

    @Test
    fun withPolicy_threads_the_filter() = runBlocking {
        val file = mixedFile()
        val delivered = mutableListOf<String>()
        SafeTensorsParametersLoader.withPolicy(
            sourceProvider = { JvmRandomAccessSource.open(file) },
            policy = DTypePolicy.Any,
            tensorFilter = { it.name != "positions" },
        ).load<FP32, Float>(DirectCpuExecutionContext(), FP32::class) { name, _ -> delivered += name }
        assertEquals(listOf("weight"), delivered)
    }
}
