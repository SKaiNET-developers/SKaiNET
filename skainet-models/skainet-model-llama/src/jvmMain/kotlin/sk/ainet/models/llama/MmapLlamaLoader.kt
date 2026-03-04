package sk.ainet.models.llama

import kotlinx.io.asSource
import kotlinx.io.buffered
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderTensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.MmapTensorSource
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.nio.file.Path
import kotlin.reflect.KClass

/**
 * Memory-mapped GGUF loader that provides zero-copy tensor access.
 *
 * This loader memory-maps the GGUF file and creates tensor views that reference
 * the mapped memory directly, avoiding data copies. This is particularly efficient
 * for large models as:
 * - Memory is loaded on-demand by the OS (lazy loading)
 * - Multiple processes can share the same mapped pages
 * - No explicit memory allocation for tensor data
 *
 * Limitations:
 * - JVM only (uses java.nio.MappedByteBuffer)
 * - Currently only supports F32 tensors (no quantized tensor support yet)
 * - Quantized models require dequantization which defeats the zero-copy benefit
 *
 * Usage:
 * ```kotlin
 * val loader = MmapLlamaLoader(path)
 * val weights = loader.loadToMap<FP32, Float>(ctx)
 * // Use weights...
 * loader.close() // Release mmap when done
 * ```
 *
 * @param filePath path to the GGUF model file
 */
public class MmapLlamaLoader(
    filePath: Path
) : AutoCloseable {

    private val fileChannel: FileChannel = FileInputStream(filePath.toFile()).channel
    private val mmapSource: MmapTensorSource = MmapTensorSource.fromChannel(fileChannel)

    // Parse header without loading tensor data
    private val reader: GGUFReader by lazy {
        FileInputStream(filePath.toFile()).use { fis ->
            GGUFReader(fis.asSource().buffered(), loadTensorData = false)
        }
    }

    private val metadata: LlamaModelMetadata by lazy {
        metadataFromGguf(reader.fields, reader.tensors)
    }

    /**
     * Load model weights as mmap-backed tensors.
     *
     * @param T the data type (must be FP32 for mmap loading)
     * @param V the value type (Float for FP32)
     * @param ctx execution context (used to wrap TensorData with ops)
     * @return LlamaWeights containing mmap-backed tensors
     */
    public fun <T : DType, V> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): LlamaWeights<T, V> {
        require(dtype == FP32::class) {
            "MmapLlamaLoader currently supports FP32 tensors only (got ${dtype.simpleName})"
        }

        validateMetadata(metadata)

        val required = requiredTensorNames(metadata)
        val tensorByName = reader.tensors.associateBy { it.name }

        val tensors = linkedMapOf<String, Tensor<T, V>>()

        // Load required tensors
        required.forEach { name ->
            val rt = tensorByName[name]
                ?: error("Missing required tensor in GGUF payload: $name")
            validateTensorShape(name, rt, metadata)
            tensors[name] = createMmapTensor(ctx, dtype, rt)
        }

        // Optional RoPE tensors (if present and F32)
        listOf(
            LlamaTensorNames.ROPE_FREQS_REAL,
            LlamaTensorNames.ROPE_FREQS_IMAG
        ).forEach { name ->
            val rt = tensorByName[name]
            if (rt != null && rt.tensorType == GGMLQuantizationType.F32) {
                tensors[name] = createMmapTensor(ctx, dtype, rt)
            }
        }

        return LlamaWeights(metadata, tensors)
    }

    public inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): LlamaWeights<T, V> = loadToMap(ctx, T::class)

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> createMmapTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        rt: ReaderTensor
    ): Tensor<T, V> {
        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())

        return when (rt.tensorType) {
            GGMLQuantizationType.F32 -> {
                // Zero-copy mmap tensor - create TensorData backed by mmap
                val mmapData = mmapSource.floatTensorAt<T>(
                    byteOffset = rt.dataOffset.toLong(),
                    shape = shape
                )
                // Use ExecutionContext.fromData to wrap with proper ops
                ctx.fromData(mmapData, dtype) as Tensor<T, V>
            }

            else -> {
                // For non-F32 tensors, fall back to streaming load with dequantization
                // This is less efficient but maintains compatibility
                error(
                    "MmapLlamaLoader only supports F32 tensors directly. " +
                    "Tensor ${rt.name} has type ${rt.tensorType}. " +
                    "Use the streaming LlamaWeightLoader with DEQUANTIZE_TO_FP32 policy for quantized models."
                )
            }
        }
    }

    override fun close() {
        mmapSource.close()
        fileChannel.close()
    }

    // Reuse validation logic from LlamaWeightLoader
    private fun metadataFromGguf(
        fields: Map<String, sk.ainet.io.gguf.ReaderField>,
        tensors: List<ReaderTensor>
    ): LlamaModelMetadata {
        val arch = fields["general.architecture"]?.stringValue() ?: "unknown"

        val embeddingLength = fields["llama.embedding_length"]?.scalarInt()
            ?: inferEmbeddingFromTensor(tensors)
        val contextLength = fields["llama.context_length"]?.scalarInt() ?: 0
        val blockCount = fields["llama.block_count"]?.scalarInt() ?: 0
        val headCount = fields["llama.attention.head_count"]?.scalarInt() ?: 0
        val kvHeadCount = fields["llama.attention.head_count_kv"]?.scalarInt() ?: headCount
        val feedForwardLength = fields["llama.feed_forward_length"]?.scalarInt() ?: 0
        val ropeDim = fields["llama.rope.dimension_count"]?.scalarInt()
        val vocabSize = fields["llama.vocab_size"]?.scalarInt()
            ?: inferVocabFromTensor(tensors)

        return LlamaModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            feedForwardLength = feedForwardLength,
            ropeDimensionCount = ropeDim,
            vocabSize = vocabSize
        )
    }

    private fun validateMetadata(metadata: LlamaModelMetadata) {
        require(metadata.architecture == "llama") {
            "Unsupported architecture: ${metadata.architecture}"
        }
        require(metadata.embeddingLength > 0) { "Invalid embedding length ${metadata.embeddingLength}" }
        require(metadata.blockCount > 0) { "Invalid block count ${metadata.blockCount}" }
        require(metadata.headCount > 0) { "Invalid head count ${metadata.headCount}" }
        require(metadata.contextLength > 0) { "Invalid context length ${metadata.contextLength}" }
        require(metadata.vocabSize > 0) { "Invalid vocab size ${metadata.vocabSize}" }
    }

    private fun requiredTensorNames(metadata: LlamaModelMetadata): List<String> {
        val names = mutableListOf<String>()
        names += LlamaTensorNames.TOKEN_EMBEDDINGS
        names += LlamaTensorNames.OUTPUT_NORM
        names += LlamaTensorNames.OUTPUT_WEIGHT

        repeat(metadata.blockCount) { layer ->
            names += LlamaTensorNames.attnNorm(layer)
            names += LlamaTensorNames.attnQ(layer)
            names += LlamaTensorNames.attnK(layer)
            names += LlamaTensorNames.attnV(layer)
            names += LlamaTensorNames.attnOut(layer)
            names += LlamaTensorNames.ffnNorm(layer)
            names += LlamaTensorNames.ffnGate(layer)
            names += LlamaTensorNames.ffnDown(layer)
            names += LlamaTensorNames.ffnUp(layer)
        }
        return names
    }

    private fun validateTensorShape(name: String, tensor: ReaderTensor, metadata: LlamaModelMetadata) {
        val dims = tensor.shape.map { it.toInt() }
        when (name) {
            LlamaTensorNames.TOKEN_EMBEDDINGS, LlamaTensorNames.OUTPUT_WEIGHT -> {
                require(dims.size == 2 && dims.contains(metadata.embeddingLength)) {
                    "Tensor $name must be [vocab, dim] shaped; got $dims"
                }
            }
            LlamaTensorNames.OUTPUT_NORM -> {
                require(dims.size == 1 && dims[0] == metadata.embeddingLength) {
                    "Tensor $name must be [${metadata.embeddingLength}] shaped; got $dims"
                }
            }
            else -> {
                // Accept other tensors without strict validation for now
            }
        }
    }

    private fun inferEmbeddingFromTensor(tensors: List<ReaderTensor>): Int {
        val emb = tensors.find { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
        return emb?.shape?.firstOrNull()?.toInt() ?: 0
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val emb = tensors.find { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
        return emb?.shape?.lastOrNull()?.toInt() ?: 0
    }

    private fun sk.ainet.io.gguf.ReaderField.stringValue(): String? {
        val bytes = parts.getOrNull(1) ?: return null
        return (bytes as? List<*>)
            ?.filterIsInstance<UByte>()
            ?.map { it.toByte() }
            ?.toByteArray()
            ?.decodeToString()
    }

    private fun sk.ainet.io.gguf.ReaderField.scalarInt(): Int? {
        val value = parts.getOrNull(0)?.getOrNull(0) ?: return null
        return when (value) {
            is UInt -> value.toInt()
            is Int -> value
            is ULong -> value.toInt()
            is Long -> value.toInt()
            else -> null
        }
    }
}
