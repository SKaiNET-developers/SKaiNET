package sk.ainet.apps.tool.quantize

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.export.GGUFWriter
import sk.ainet.io.gguf.export.GgufTensorEntry
import sk.ainet.io.gguf.export.GgufWriteRequest
import sk.ainet.io.safetensors.NarrowFloatLoadPolicy
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Requantizers
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import java.nio.file.Files

/**
 * `skainet-quantize`: a SafeTensors checkpoint → a GGUF with its weight matrices quantized (SKEEP-007 §T2).
 *
 * The engine can quantize at load (`WeightForm(RequantizeTo(Q8_0))` on the SafeTensors loaders), but a
 * browser still has to *download* the file first, and F16 is twice the bytes of Q8_0. This tool runs the
 * same quantizers once, offline, and writes a GGUF the engine's streaming loader serves packed on every
 * target. Tensor names are kept exactly as the SafeTensors file has them, so a family's name resolver
 * works unchanged; tensors matched by `--keep` (name prefixes) stay dense, as F16 by default — the
 * usual choice for a token-embedding table that an embedding layer reads row by row.
 *
 * Usage:
 *   skainet-quantize --in model.safetensors --out model.q8_0.gguf [--encoding q8_0|q4_0] [--keep tok_embeddings,lm_head]
 */
public fun main(args: Array<String>) {
    val parser = ArgParser("skainet-quantize")
    val input by parser.option(ArgType.String, fullName = "in", shortName = "i", description = "SafeTensors file to read")
    val output by parser.option(ArgType.String, fullName = "out", shortName = "o", description = "GGUF file to write")
    val encodingName by parser.option(ArgType.String, fullName = "encoding", shortName = "e", description = "Target block encoding: q8_0 (default) or q4_0").default("q8_0")
    val keep by parser.option(ArgType.String, fullName = "keep", shortName = "k", description = "Comma-separated tensor-name prefixes kept dense (F16)").default("")
    val keepF32 by parser.option(ArgType.Boolean, fullName = "keep-f32", description = "Write kept float tensors as F32 instead of F16").default(false)
    parser.parse(args)
    require(!input.isNullOrBlank()) { "--in is required" }
    require(!output.isNullOrBlank()) { "--out is required" }
    val report = quantizeSafeTensorsToGguf(
        input = input!!,
        output = output!!,
        encoding = parseEncoding(encodingName),
        keepPrefixes = keep.split(',').map { it.trim() }.filter { it.isNotEmpty() },
        keepAsF32 = keepF32,
    )
    println(report.render())
}

public fun parseEncoding(name: String): TensorEncoding = when (name.lowercase()) {
    "q8_0" -> TensorEncoding.Q8_0
    "q4_0" -> TensorEncoding.Q4_0
    else -> throw IllegalArgumentException("unknown encoding '$name'; quantize-at-load supports ${Requantizers.supported.joinToString { it.name.lowercase() }}")
}

public data class QuantizeReport(
    val encoding: TensorEncoding,
    val quantized: List<String>,
    val kept: List<String>,
    val inputBytes: Long,
    val outputBytes: Long,
) {
    public fun render(): String = buildString {
        appendLine("skainet-quantize: ${quantized.size} tensors → ${encoding.name}, ${kept.size} kept dense")
        appendLine("  input  ${inputBytes / (1024 * 1024)} MB")
        appendLine("  output ${outputBytes / (1024 * 1024)} MB")
        if (kept.isNotEmpty()) appendLine("  kept: ${kept.joinToString()}")
    }
}

/**
 * Quantize every rank-2 float tensor of [input] whose rows block by the encoding's block size to
 * [encoding], keep the rest dense, and write a GGUF to [output]. Returns what happened.
 */
public fun quantizeSafeTensorsToGguf(
    input: String,
    output: String,
    encoding: TensorEncoding = TensorEncoding.Q8_0,
    keepPrefixes: List<String> = emptyList(),
    keepAsF32: Boolean = false,
): QuantizeReport {
    require(Requantizers.supports(encoding)) { "no quantizer for ${encoding.name}" }
    val requantize = WeightForm(encoding = EncodingRequest.RequantizeTo(encoding), shape = WeightShapeOrientation.OUT_IN)
    fun kept(name: String) = keepPrefixes.any { name.startsWith(it) }

    val ctx = DefaultDataExecutionContext()
    val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
    runBlocking {
        SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(input) },
            // Kept narrow floats stay narrow so they can be written back as F16/BF16 bytes verbatim.
            bf16Policy = NarrowFloatLoadPolicy.KEEP_NATIVE,
            fp16Policy = NarrowFloatLoadPolicy.KEEP_NATIVE,
            weightFormFor = { name -> if (kept(name)) null else requantize },
        ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
    }

    val quantized = ArrayList<String>()
    val keptNames = ArrayList<String>()
    val entries = loaded.map { (name, tensor) ->
        val dims = tensor.shape.dimensions.toList()
        // What to write is read from the view's format (SKEEP-003): block-packed bytes go out as
        // they are, kept 16-bit floats too, everything else is written from its values.
        val view = tensor.data.view
        val raw = view?.let { wholeHeapBytes(it) }
        val narrow = view != null && raw != null && !view.layout.blocked &&
            view.format.encoding == TensorEncoding.Dense(2) && view.isContiguous
        when {
            view != null && view.layout.blocked -> {
                quantized += name
                val bytes = raw ?: error("tensor '$name': packed ${view.format.encoding?.name} bytes are not one whole heap array")
                GgufTensorEntry(ggufName = name, quantization = ggmlTypeOf(checkNotNull(view.format.encoding)), shape = dims, rawBytes = bytes)
            }
            narrow && view!!.format.dtype == FP16 -> {
                keptNames += name
                if (keepAsF32) GgufTensorEntry(ggufName = name, tensor = tensor, quantization = GGMLQuantizationType.F32, shape = dims)
                else GgufTensorEntry(ggufName = name, quantization = GGMLQuantizationType.F16, shape = dims, rawBytes = raw)
            }
            narrow && view!!.format.dtype == BF16 -> {
                keptNames += name
                if (keepAsF32) GgufTensorEntry(ggufName = name, tensor = tensor, quantization = GGMLQuantizationType.F32, shape = dims)
                else GgufTensorEntry(ggufName = name, quantization = GGMLQuantizationType.BF16, shape = dims, rawBytes = raw)
            }
            else -> {
                keptNames += name
                val type = if (keepAsF32) GGMLQuantizationType.F32 else GGMLQuantizationType.F16
                GgufTensorEntry(ggufName = name, tensor = tensor, quantization = type, shape = dims)
            }
        }
    }
    val metadata = linkedMapOf<String, Any>(
        "general.name" to input.substringAfterLast('/').substringBeforeLast('.'),
        "skainet.quantize.source" to "safetensors",
        "skainet.quantize.encoding" to encoding.name,
    )
    val request = GgufWriteRequest(metadata = metadata, tensors = entries, tensorMap = entries.associate { it.ggufName to it.ggufName })
    val outPath = Path(output)
    SystemFileSystem.sink(outPath).buffered().use { sink -> GGUFWriter.writeToSink(request, sink) }
    return QuantizeReport(
        encoding = encoding,
        quantized = quantized,
        kept = keptNames,
        inputBytes = Files.size(java.nio.file.Path.of(input)),
        outputBytes = SystemFileSystem.metadataOrNull(outPath)?.size ?: -1L,
    )
}

private fun ggmlTypeOf(encoding: TensorEncoding): GGMLQuantizationType = when (encoding) {
    TensorEncoding.Q8_0 -> GGMLQuantizationType.Q8_0
    TensorEncoding.Q4_0 -> GGMLQuantizationType.Q4_0
    else -> throw IllegalArgumentException("${encoding.name} has no GGUF tensor type")
}

/** The byte array behind [view] when the view starts at its first byte and the array holds nothing else. */
private fun wholeHeapBytes(view: TensorView): ByteArray? {
    val heap = view.storage as? Storage.Heap ?: return null
    val bytes = heap.bytes ?: return null
    if (heap.arrayOffset != 0 || view.layout.offsetElements != 0L || heap.sizeBytes != bytes.size.toLong()) return null
    return bytes
}
