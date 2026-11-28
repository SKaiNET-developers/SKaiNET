package sk.ainet.docs.samples

// tag::imports[]
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.frontend.tokenizer.TextFrontEnds
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.io.gguf.export.GGUFWriter
import sk.ainet.io.gguf.export.GgufTensorEntry
import sk.ainet.io.gguf.export.GgufWriteRequest
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
// end::imports[]
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The samples behind `explanation/gguf-handling.adoc`: a toy GGUF file is written, inspected
 * without touching its tensor bytes, loaded through the streaming loader, and asked for its
 * tokenizer. Every tagged region is included verbatim into the page; `GgufHandlingTest` runs it.
 */
object GgufHandling {

    const val ROWS: Int = 4
    const val COLS: Int = 32

    // tag::write[]
    /**
     * A toy model file: an FP32 embedding, a Q8_0 weight (34 bytes per 32 values; all-zero
     * blocks decode to 0.0) and the metadata a GGUF reader cares about, including a minimal
     * byte-level BPE vocabulary under `tokenizer.ggml.*`.
     */
    fun writeToyModel(file: File) {
        val embedding = FloatArray(ROWS * COLS) { i -> i * 0.5f }
        val f32Bytes = ByteBuffer.allocate(embedding.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            .apply { embedding.forEach { putFloat(it) } }.array()
        val q80Bytes = ByteArray(ROWS * COLS / 32 * 34)

        val request = GgufWriteRequest(
            metadata = mapOf(
                "general.architecture" to "toy",
                "general.name" to "toy-model",
                "tokenizer.ggml.model" to "gpt2",
                "tokenizer.ggml.tokens" to listOf("h", "i", "hi"),
                "tokenizer.ggml.merges" to listOf("h i"),
            ),
            tensors = listOf(
                GgufTensorEntry(
                    ggufName = "tok_embeddings.weight",
                    quantization = GGMLQuantizationType.F32,
                    shape = listOf(ROWS, COLS),
                    rawBytes = f32Bytes,
                ),
                GgufTensorEntry(
                    ggufName = "blk.0.ffn.weight",
                    quantization = GGMLQuantizationType.Q8_0,
                    shape = listOf(ROWS, COLS),
                    rawBytes = q80Bytes,
                ),
            ),
            tensorMap = emptyMap(),
        )
        val (_, bytes) = GGUFWriter.writeToByteArray(request)
        file.writeBytes(bytes)
    }
    // end::write[]

    // tag::inspect[]
    /** What a metadata-only pass sees: header numbers, KV fields and the tensor directory. */
    class FileSummary(
        val version: UInt,
        val tensorCount: ULong,
        val architecture: String?,
        val tensors: List<String>,
        val bytesByType: Map<String, Long>,
    )

    /** Reads everything about the file except its tensor bytes. */
    fun inspect(file: File): FileSummary =
        StreamingGGUFReader.open(JvmRandomAccessSource.open(file)).use { reader ->
            FileSummary(
                version = reader.version,
                tensorCount = reader.tensorCount,
                architecture = reader.fields["general.architecture"] as? String,
                tensors = reader.tensors.map { t ->
                    "${t.name}: ${t.shape} ${t.tensorType} (${t.nBytes} bytes)"
                },
                bytesByType = reader.tensors
                    .groupBy { it.tensorType.name }
                    .mapValues { (_, infos) -> infos.sumOf { it.nBytes } },
            )
        }
    // end::inspect[]

    // tag::load[]
    /** Streams every tensor into memory; quantized types arrive packed or widened to FP32. */
    suspend fun loadAll(file: File): Map<String, Tensor<FP32, Float>> {
        val loaded = mutableMapOf<String, Tensor<FP32, Float>>()
        val loader = StreamingGgufParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) },
        )
        loader.load<FP32, Float>(DefaultDataExecutionContext(), FP32::class) { name, tensor ->
            loaded[name] = tensor
        }
        return loaded
    }
    // end::load[]

    // tag::vocab[]
    /** Builds the tokenizer the file ships in its own metadata and round-trips one word. */
    fun tokenizerRoundTrip(file: File): Pair<List<Int>, String> =
        StreamingGGUFReader.open(JvmRandomAccessSource.open(file)).use { reader ->
            val tokenizer = TextFrontEnds.fromGguf(reader.fields)
            val ids = tokenizer.encode("hi")
            ids.toList() to tokenizer.decode(ids)
        }
    // end::vocab[]
}
