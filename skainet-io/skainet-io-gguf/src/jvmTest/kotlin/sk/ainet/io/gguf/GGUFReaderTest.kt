package sk.ainet.io.gguf

import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Assert
import org.junit.Test
import sk.ainet.io.gguf.utils.reshape
import kotlin.collections.iterator
import kotlin.test.assertNotNull

class GGUFReaderTest {

    @Test
    fun test_reshape() {
        Assert.assertEquals(
            listOf(listOf(1, 2), listOf(3, 4)),
            listOf(1, 2, 3, 4).reshape(2, 2)
        )

        Assert.assertEquals(
            listOf(listOf(1, 2, 3, 4)),
            listOf(1, 2, 3, 4).reshape(1, 4)
        )

        Assert.assertEquals(
            listOf(listOf(1), listOf(2), listOf(3), listOf(4)),
            listOf(1, 2, 3, 4).reshape(4, 1)
        )

        Assert.assertNotSame(
            listOf(listOf(1), listOf(2), listOf(3), listOf(4)),
            List(384 * 1536) { it + 0.02387242 }.reshape(384, 1536)
        )
    }


    /**
     * You may run this test to read from a gguf model and print model metadata in console output.
     * Change the file path to read from your own model.
     *
     * To find a gguf model, you can look from here:
     * https://huggingface.co/models?library=gguf
     */
    @Test
    fun `test GGUFReader with valid model file`() {
        javaClass.getResourceAsStream("/test_experiment.gguf").use { inputStream ->
            assertNotNull(inputStream, "Test resource file not found!")

            // Convert it to a RawSource and then buffer it to get a Source:

            val reader = GGUFReader(inputStream.asSource().buffered())
            // List all key-value pairs in a columned format
            println("Key-Value Pairs:")
            val maxKeyLength = reader.fields.keys.maxOf { it.length }
            for ((key, field) in reader.fields) {
                val value = if (field.types[0] == GGUFValueType.STRING && field.types.size == 1) {
                    val idx = field.data.firstOrNull() ?: -1
                    val part = field.parts.getOrNull(idx)
                    val bytes = (part as? List<*>)?.mapNotNull { (it as? Number)?.toInt()?.toByte() }?.toByteArray()
                    bytes?.let { String(it, Charsets.UTF_8) } ?: part ?: ""
                } else {
                    val idx = field.data.firstOrNull()?.takeIf { it >= 0 && it < field.parts.size }
                    if (idx != null) field.parts[idx] else field.parts
                }

                println("${key.padEnd(maxKeyLength)} : $value")
            }
            println("----")

            // List all tensors
            println("Tensors:")
            val tensorInfoFormat = "%-30s | Shape: %-15s | Size: %-12s | Quantization: %s"
            println(tensorInfoFormat.format("Tensor Name", "Shape", "Size", "Quantization"))
            println("-".repeat(80))
            for (tensor in reader.tensors) {
                val shapeStr = tensor.shape.joinToString("x")
                val sizeStr = tensor.nElements.toString()
                val quantizationStr = tensor.tensorType.name
                println(tensorInfoFormat.format(tensor.name, shapeStr, sizeStr, quantizationStr))
            }
        }
    }
}
