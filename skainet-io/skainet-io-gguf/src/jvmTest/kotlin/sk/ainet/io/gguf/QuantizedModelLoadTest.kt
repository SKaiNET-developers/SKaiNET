package sk.ainet.io.gguf

import org.junit.Test
import sk.ainet.io.JvmRandomAccessSource
import java.io.File
import kotlin.test.assertTrue
import kotlin.test.assertEquals

/**
 * Integration test to verify quantized GGUF model loading works correctly
 * after the tensor size calculation fix.
 */
class QuantizedModelLoadTest {

    @Test
    fun `verify Q8_0 model tensor sizes are calculated correctly`() {
        val modelPath = "models/tinyllama-1.1b-q8.gguf"
        // Navigate from skainet-io/skainet-io-gguf up to project root
        var projectRoot = File(System.getProperty("user.dir"))
        while (!projectRoot.resolve("models").exists() && projectRoot.parentFile != null) {
            projectRoot = projectRoot.parentFile
        }
        val modelFile = projectRoot.resolve(modelPath)

        if (!modelFile.exists()) {
            println("Skipping test: Model file not found at ${modelFile.absolutePath}")
            return
        }

        println("Testing with model: ${modelFile.absolutePath}")

        JvmRandomAccessSource.open(modelFile).use { source ->
            val reader = StreamingGGUFReader.open(source)

            println("GGUF Version: ${reader.version}")
            println("Tensor count: ${reader.tensorCount}")
            println("Architecture: ${reader.fields["general.architecture"]}")

            // Verify we can read tensor metadata
            assertTrue(reader.tensors.isNotEmpty(), "Should have tensors")

            // Check first few tensors
            reader.tensors.take(5).forEach { tensor ->
                println("Tensor: ${tensor.name}")
                println("  Shape: ${tensor.shape}")
                println("  Type: ${tensor.tensorType}")
                println("  Elements: ${tensor.nElements}")
                println("  Bytes: ${tensor.nBytes}")

                // Verify nBytes calculation: (nElements / blockSize) * typeSize
                val (blockSize, typeSize) = GGML_QUANT_SIZES[tensor.tensorType]!!
                val expectedBytes = (tensor.nElements.toLong() / blockSize) * typeSize
                assertEquals(expectedBytes, tensor.nBytes,
                    "Tensor ${tensor.name}: nBytes mismatch - expected $expectedBytes, got ${tensor.nBytes}")
            }

            // Try to load one small tensor's data to verify the byte count is correct
            val smallTensor = reader.tensors.minByOrNull { it.nBytes }
            if (smallTensor != null) {
                println("\nLoading smallest tensor: ${smallTensor.name} (${smallTensor.nBytes} bytes)")
                val data = reader.loadTensorData(smallTensor)
                assertEquals(smallTensor.nBytes, data.size.toLong(),
                    "Loaded data size should match nBytes")
                println("Successfully loaded ${data.size} bytes!")
            }

            println("\n✓ All tensor size calculations verified!")
        }
    }
}
