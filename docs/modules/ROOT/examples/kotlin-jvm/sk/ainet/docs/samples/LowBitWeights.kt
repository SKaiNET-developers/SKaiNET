package sk.ainet.docs.samples

// tag::imports[]
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.export.GGUFWriter
import sk.ainet.io.gguf.export.GgufTensorEntry
import sk.ainet.io.gguf.export.GgufWriteRequest
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.MemoryTracker
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
// end::imports[]
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sin

/**
 * SKEEP-003 walk-through: one weight stored six ways in a GGUF file, read back through one type,
 * [TensorView]. Every tagged region is included verbatim into
 * `tutorials/low-bit-weights-getting-started.adoc`; `LowBitWeightsTest` runs it.
 */
object LowBitWeights {

    /** A `[ROWS, COLS]` weight: 8 outputs, 256 inputs. 256 is the block size of the ternary formats. */
    const val ROWS: Int = 8
    const val COLS: Int = 256

    /** What one stored form of the weight costs and how far it is from the trained values. */
    class FormatLine(
        val name: String,
        val format: String,
        val logicalBytes: Long,
        val physicalBytes: Long,
        val bitsPerWeight: Double,
        val maxError: Float,
    )

    class Result(
        val lines: List<FormatLine>,
        val totalLogicalBytes: Long,
        val totalPhysicalBytes: Long,
        val mappedDomain: String,
        val mappedEqualsHeap: Boolean,
        val materializedFormat: String,
        val materializeIsLossless: Boolean,
        val requantizedFormat: String,
        val requantizeError: Float,
        val kernel: String,
        val product: FloatArray,
        val reference: FloatArray,
    )

    // tag::weight[]
    /** Stands in for a trained FP32 weight: smooth values in [-1, 1]. */
    fun trainedWeight(): FloatArray = FloatArray(ROWS * COLS) { i -> sin(0.37f * i) * 0.8f }
    // end::weight[]

    // tag::write[]
    /** The same weight six ways in one GGUF file. */
    fun writeModel(file: File, weight: FloatArray) {
        val shape = listOf(ROWS, COLS)
        // A view over the trained floats. `requantizeTo` is the lossy direction: it returns a view
        // over new packed bytes.
        val dense = TensorView.dense(Storage.Heap.wrap(weight), Shape(ROWS, COLS), FP32)

        fun entry(name: String, type: GGMLQuantizationType, bytes: ByteArray) =
            GgufTensorEntry(ggufName = name, quantization = type, shape = shape, rawBytes = bytes)

        val tensors = listOf(
            entry("w.f32", GGMLQuantizationType.F32, floatBytes(weight)),
            entry("w.f16", GGMLQuantizationType.F16, halfBytes(weight)),
            entry("w.q8_0", GGMLQuantizationType.Q8_0, bytesOf(dense.requantizeTo(TensorEncoding.Q8_0))),
            entry("w.q4_0", GGMLQuantizationType.Q4_0, bytesOf(dense.requantizeTo(TensorEncoding.Q4_0))),
            entry("w.tq2_0", GGMLQuantizationType.TQ2_0, TernaryCodec.encodeTq2_0(weight)),
            entry("w.tq1_0", GGMLQuantizationType.TQ1_0, TernaryCodec.encodeTq1_0(weight)),
        )
        val request = GgufWriteRequest(
            metadata = mapOf("general.architecture" to "tutorial"),
            tensors = tensors,
            tensorMap = tensors.associate { it.ggufName to it.ggufName },
        )
        val (_, bytes) = GGUFWriter.writeToByteArray(request)
        file.writeBytes(bytes)
    }

    /** The bytes a view's storage holds. */
    fun bytesOf(view: TensorView): ByteArray =
        ByteArray(view.storage.sizeBytes.toInt()).also { view.storage.copyInto(it) }
    // end::write[]

    private fun floatBytes(values: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buffer.putFloat(it) }
        return buffer.array()
    }

    private fun halfBytes(values: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(values.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buffer.putShort(Fp16Codec.encode(it).toShort()) }
        return buffer.array()
    }

    private fun maxError(a: FloatArray, b: FloatArray): Float {
        var max = 0f
        for (i in a.indices) max = maxOf(max, abs(a[i] - b[i]))
        return max
    }

    fun run(file: File): Result {
        val weight = trainedWeight()
        writeModel(file, weight)

        val tracker = MemoryTracker()
        val lines = ArrayList<FormatLine>()
        val result: Result

        JvmRandomAccessSource.open(file).use { source ->
            val reader = StreamingGGUFReader.open(source)

            // tag::read[]
            for (info in reader.tensors) {
                // One call and one type for every format. Nothing is decoded here: the view holds
                // the file's bytes, and `view.format` says what they are.
                val view: TensorView = reader.loadTensorView(info)

                val decoded = view.toFloatArray()       // decodes on read, whatever the format
                val first = view.get(0, 1)              // one element, the same call for all six
                check(first == decoded[1])

                tracker.record(info.name, view)         // logical against physical bytes
                // end::read[]
                val physical = view.format.physicalBytes(view.elementCount) ?: view.storage.sizeBytes
                lines += FormatLine(
                    name = info.name,
                    format = view.format.toString(),
                    logicalBytes = view.elementCount * view.format.dtype.sizeInBytes,
                    physicalBytes = physical,
                    bitsPerWeight = physical * 8.0 / view.elementCount,
                    maxError = maxError(decoded, weight),
                )
                // tag::read[]
            }
            val report = tracker.report()
            // report.totalLogicalBytes: what the tensors are worth as their dtype
            // report.totalPhysicalBytes: what they occupy
            // end::read[]

            // tag::mapped[]
            // The same view over the mapped file: no heap copy, and the same code reads it.
            val info = reader.tensors.single { it.name == "w.tq2_0" }
            val heap = reader.loadTensorView(info)
            val mapped = reader.loadTensorViewMapped(info, file.path)
            val mappedDomain = mapped.storage.domain.name               // MMAP_FILE
            val mappedEqualsHeap = mapped.toFloatArray().contentEquals(heap.toFloatArray())
            mapped.storage.close()                                      // the caller unmaps
            // end::mapped[]

            // tag::convert[]
            // Lossless: packed to dense FP32. The values are the ones the packed bytes decode to.
            val materialized = heap.materialize()
            val lossless = materialized.toFloatArray().contentEquals(heap.toFloatArray())

            // Lossy, and it says so in its name: dense FP32 to a packed encoding.
            val f32 = reader.loadTensorView("w.f32")
            val q4 = f32.requantizeTo(TensorEncoding.Q4_0)
            val requantizeError = maxError(q4.toFloatArray(), f32.toFloatArray())
            // end::convert[]

            // tag::matmul[]
            // x · Wᵀ on the packed Q4_0 weight. The kernel is looked up by the formats of the two
            // views and reads the packed blocks: no dense copy of the weight is made. The trace
            // names the kernel that ran.
            val weightQ4 = reader.loadTensorView("w.q4_0")
            val x = FloatArray(COLS) { i -> ((i % 7) - 3) / 3f }
            val activation = TensorView.dense(Storage.Heap.wrap(x), Shape(1, COLS), FP32)
            val out = TensorView.dense(Storage.Heap.floats(ROWS), Shape(1, ROWS), FP32)
            val sink = RecordingTraceSink()
            // `prepackWeights`: put the file-order blocks into the order the kernel feeds on, for
            // this one call. A model does that once at load with `view.prepack(...)`.
            KernelDispatch.matmul(activation, weightQ4, out, Scope.Ambient, sink, prepackWeights = true)
            val kernel = sink.eventsOf<TraceEvent.KernelRun>().single().kernel
            val product = FloatArray(ROWS) { out.get(0, it) }
            // end::matmul[]

            val w = weightQ4.toFloatArray()
            val reference = FloatArray(ROWS) { r ->
                var sum = 0f
                for (c in 0 until COLS) sum += x[c] * w[r * COLS + c]
                sum
            }

            result = Result(
                lines = lines.sortedByDescending { it.bitsPerWeight },   // the writer orders tensors by name
                totalLogicalBytes = report.totalLogicalBytes,
                totalPhysicalBytes = report.totalPhysicalBytes,
                mappedDomain = mappedDomain,
                mappedEqualsHeap = mappedEqualsHeap,
                materializedFormat = materialized.format.toString(),
                materializeIsLossless = lossless,
                requantizedFormat = q4.format.toString(),
                requantizeError = requantizeError,
                kernel = kernel,
                product = product,
                reference = reference,
            )
        }
        return result
    }

    /** The table the tutorial shows, as text. */
    fun render(result: Result): String = buildString {
        appendLine("name      format                 bytes   bits/weight   max error")
        for (l in result.lines) {
            appendLine(
                "${l.name.padEnd(9)} ${l.format.padEnd(20)} ${l.physicalBytes.toString().padStart(7)}   " +
                    "${"%.3f".format(l.bitsPerWeight).padStart(11)}   ${"%.4f".format(l.maxError)}"
            )
        }
        appendLine("total: ${result.totalPhysicalBytes} bytes stored, ${result.totalLogicalBytes} bytes as their dtype")
        appendLine("mapped: ${result.mappedDomain}, equal to the heap read: ${result.mappedEqualsHeap}")
        appendLine("materialize -> ${result.materializedFormat}, lossless: ${result.materializeIsLossless}")
        appendLine("requantizeTo -> ${result.requantizedFormat}, max error ${"%.4f".format(result.requantizeError)}")
        appendLine("matmul kernel: ${result.kernel}")
    }
}
