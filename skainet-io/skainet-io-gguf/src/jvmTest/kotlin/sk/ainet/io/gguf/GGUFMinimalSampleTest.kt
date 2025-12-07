package sk.ainet.io.gguf

import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Minimal sample demonstrating how to:
 * - load a GGUF file
 * - read only metadata without loading entire tensor payloads
 * - lazily materialize tensors and create DSL tensors respecting dtype
 * - build a name->tensor map (using ExecutionContext)
 */
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8

class GGUFMinimalSampleTest {

    private fun ggmlToDType(q: GGMLQuantizationType): Class<out DType> = when (q) {
        // Map native types. Note: FP16 exists but DSL stores it as a distinct type; we avoid it here.
        GGMLQuantizationType.F32 -> FP32::class.java
        GGMLQuantizationType.I8 -> Int8::class.java
        GGMLQuantizationType.I32 -> Int32::class.java
        // For unsupported/native-or-quantized types in this minimal sample, we fallback to byte storage (Int8)
        GGMLQuantizationType.F16 -> FP16::class.java
        GGMLQuantizationType.F64, GGMLQuantizationType.I16, GGMLQuantizationType.I64 -> Int8::class.java
        else -> Int8::class.java
    }

    @Test
    fun minimal_metadata_only_and_lazy_tensor_creation() {
        javaClass.getResourceAsStream("/test_experiment.gguf").use { inputStream ->
            assertNotNull(inputStream, "Test resource file not found!")

            // 1) Load GGUF reading only metadata (no tensor payloads yet)
            val reader = GGUFReader(inputStream.asSource().buffered(), loadTensorData = false)

            // Header/meta fields are available
            val versionField = reader.fields["GGUF.version"]
            assertNotNull(versionField)
            val tensorCountField = reader.fields["GGUF.tensor_count"]
            assertNotNull(tensorCountField)

            // 2) Tensors are discovered, but no payloads loaded
            assertTrue(reader.tensors.isNotEmpty(), "No tensors discovered in the GGUF file")
            assertTrue(reader.tensors.all { it.data.isEmpty() }, "Tensor payloads should not be loaded in metadata-only mode")

            // 3) Build a name->ReaderTensor map (metadata only)
            val metaTensorByName: Map<String, ReaderTensor> = reader.tensors.associateBy { it.name }
            assertTrue(metaTensorByName.isNotEmpty())

            // 4) Lazily materialize a couple of tensors and create DSL tensors with an ExecutionContext
            val ctx = DefaultDataExecutionContext()

            // Pick up to first 3 tensors to demonstrate
            val sample = reader.tensors.take(3)
            val created = mutableMapOf<String, Any>()

            for (rt in sample) {
                val raw = reader.materialize(rt)
                val dtypeCls = ggmlToDType(rt.tensorType)

                // Create a DSL tensor according to dtype mapping
                val shape = Shape(rt.shape.map { it.toInt() }.toIntArray())
                val dslTensor: Any = when (rt.tensorType) {
                    GGMLQuantizationType.F32 -> ctx.fromFloatArray<FP32, Float>(
                        shape,
                        FP32::class,
                        raw.map { (it as Number).toFloat() }.toFloatArray()
                    )
                    GGMLQuantizationType.I32 -> ctx.fromIntArray<Int32, Int>(
                        shape,
                        Int32::class,
                        raw.map { (it as Number).toInt() }.toIntArray()
                    )
                    GGMLQuantizationType.I8 -> ctx.fromByteArray<Int8, Byte>(
                        shape,
                        Int8::class,
                        raw.map { (it as Number).toByte() }.toByteArray()
                    )
                    // All other quantized/native types: keep raw bytes as Int8 tensor for this minimal sample
                    else -> ctx.fromByteArray<Int8, Byte>(
                        shape,
                        Int8::class,
                        raw.map { (it as Number).toByte() }.toByteArray()
                    )
                }
                created[rt.name] = dslTensor
            }

            // 5) Build a names hashmap with created DSL tensors (subset)
            val nameToDslTensor = created.toMap()
            assertEquals(sample.size, nameToDslTensor.size)

            // For demonstration, print a tiny summary
            println("Metadata: version=${versionField.parts.last().first()} tensorCount=${tensorCountField.parts.last().first()}")
            println("Sample tensors created:")
            nameToDslTensor.forEach { (name, tensor) ->
                println("- $name -> ${tensor::class.simpleName}")
            }
        }
    }
}
