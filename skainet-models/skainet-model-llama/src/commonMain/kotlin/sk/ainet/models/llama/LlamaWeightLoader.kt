package sk.ainet.models.llama

import kotlinx.io.Source
import kotlinx.io.buffered
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderField
import sk.ainet.io.gguf.QK_K
import sk.ainet.io.gguf.ReaderTensor
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingTensorInfo
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.math.pow
import kotlin.math.max
import kotlin.ExperimentalUnsignedTypes
import kotlin.reflect.KClass

public data class LlamaModelMetadata(
    val architecture: String,
    val embeddingLength: Int,
    val contextLength: Int,
    val blockCount: Int,
    val headCount: Int,
    val kvHeadCount: Int,
    val feedForwardLength: Int,
    val ropeDimensionCount: Int?,
    val vocabSize: Int
)

public data class LlamaWeights<T : DType, V>(
    val metadata: LlamaModelMetadata,
    val tensors: Map<String, Tensor<T, V>>,
    val quantTypes: Map<String, GGMLQuantizationType> = emptyMap()
)

public object LlamaTensorNames {
    const val TOKEN_EMBEDDINGS: String = "token_embd.weight"
    const val OUTPUT_NORM: String = "output_norm.weight"
    const val OUTPUT_WEIGHT: String = "output.weight"
    const val ROPE_FREQS_REAL: String = "rope.freq_cis_real"
    const val ROPE_FREQS_IMAG: String = "rope.freq_cis_imag"

    fun attnNorm(layer: Int): String = "blk.$layer.attn_norm.weight"
    fun attnQ(layer: Int): String = "blk.$layer.attn_q.weight"
    fun attnK(layer: Int): String = "blk.$layer.attn_k.weight"
    fun attnV(layer: Int): String = "blk.$layer.attn_v.weight"
    fun attnOut(layer: Int): String = "blk.$layer.attn_output.weight"
    fun ffnNorm(layer: Int): String = "blk.$layer.ffn_norm.weight"
    fun ffnGate(layer: Int): String = "blk.$layer.ffn_gate.weight"
    fun ffnDown(layer: Int): String = "blk.$layer.ffn_down.weight"
    fun ffnUp(layer: Int): String = "blk.$layer.ffn_up.weight"
}

/**
 * Adapter that loads LLaMA weights from GGUF files and emits them in the canonical GGUF tensor
 * naming scheme. Validation covers metadata presence and basic shape consistency for the tensors
 * we materialize.
 */
public class LlamaWeightLoader private constructor(
    private val sourceProvider: (() -> Source)?,
    private val randomAccessProvider: (() -> RandomAccessSource)?,
    private val loadTensorData: Boolean = true,
    private val quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
    // Note: set loadTensorData=false to only validate metadata; tensors will be materialized
    // lazily when needed.
) {
    /**
     * Primary constructor for sequential Source-based loading.
     * Loads entire file into memory - suitable for models under 2GB.
     */
    public constructor(
        sourceProvider: () -> Source,
        loadTensorData: Boolean = true,
        quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
    ) : this(
        sourceProvider = sourceProvider,
        randomAccessProvider = null,
        loadTensorData = loadTensorData,
        quantPolicy = quantPolicy
    )

    /**
     * Secondary constructor for streaming RandomAccessSource-based loading.
     * Parses metadata only (~1MB memory) and loads tensors on-demand.
     * Suitable for models of any size (100+ GB).
     */
    public constructor(
        randomAccessProvider: () -> RandomAccessSource,
        quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
    ) : this(
        sourceProvider = null,
        randomAccessProvider = randomAccessProvider,
        loadTensorData = true,  // Ignored for streaming
        quantPolicy = quantPolicy
    )

    /**
     * Backward-compatible companion delegating to shared [DequantOps].
     * Existing callers (e.g. `LlamaWeightLoader.dequantF16(raw)`) continue to work.
     */
    public companion object Dequant {
        internal fun transposeColumnMajorToRowMajor(data: FloatArray, rows: Int, cols: Int): FloatArray =
            DequantOps.transposeColumnMajorToRowMajor(data, rows, cols)

        internal fun dequantF16(raw: List<Any>): FloatArray = DequantOps.dequantF16(raw)
        internal fun dequantBF16(raw: List<Any>): FloatArray = DequantOps.dequantBF16(raw)
        internal fun dequantQ4_0(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ4_0(raw, nElems)
        internal fun dequantQ5_0(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ5_0(raw, nElems)
        internal fun dequantQ8_0(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ8_0(raw, nElems)
        internal fun dequantQ4_1(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ4_1(raw, nElems)
        internal fun dequantQ5_1(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ5_1(raw, nElems)
        internal fun dequantQ8_1(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ8_1(raw, nElems)
        internal fun dequantIQ4NL(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantIQ4NL(raw, nElems)
        internal fun dequantIQ4XS(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantIQ4XS(raw, nElems)
        internal fun dequantQ2K(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ2K(raw, nElems)
        internal fun dequantQ3K(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ3K(raw, nElems)
        internal fun dequantQ4K(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ4K(raw, nElems)
        internal fun dequantQ5K(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ5K(raw, nElems)
        internal fun dequantQ6K(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ6K(raw, nElems)
        internal fun dequantQ8K(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantQ8K(raw, nElems)
        internal fun dequantTQ2_0(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantTQ2_0(raw, nElems)
        internal fun dequantTQ1_0(raw: List<Any>, nElems: Int): FloatArray = DequantOps.dequantTQ1_0(raw, nElems)
    }

    /**
     * Load weights and invoke [onTensorLoaded] for each required tensor. Returns parsed metadata.
     */
    public suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): LlamaModelMetadata {
        return loadFromGguf(ctx, dtype, onTensorLoaded, null)
    }

    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): LlamaModelMetadata = load(ctx, T::class, onTensorLoaded)

    /** Convenience helper that collects tensors into a map alongside metadata. */
    public suspend fun <T : DType, V> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): LlamaWeights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val quantTypes = linkedMapOf<String, GGMLQuantizationType>()
        val meta = loadFromGguf(ctx, dtype, { name, tensor -> byName[name] = tensor }) { name, qt ->
            quantTypes[name] = qt
        }
        return LlamaWeights(meta, byName, quantTypes)
    }

    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): LlamaWeights<T, V> = loadToMap(ctx, T::class)

    // ============== Streaming API (for large files >2GB) ==============

    /**
     * Load weights using streaming API - parses metadata only, loads tensors on-demand.
     * Requires [randomAccessProvider] constructor.
     */
    public suspend fun <T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): LlamaModelMetadata {
        return loadFromStreamingGguf(ctx, dtype, onTensorLoaded, null)
    }

    public suspend inline fun <reified T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): LlamaModelMetadata = loadStreaming(ctx, T::class, onTensorLoaded)

    /**
     * Load weights to map using streaming API.
     * Requires [randomAccessProvider] constructor.
     */
    public suspend fun <T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): LlamaWeights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val quantTypes = linkedMapOf<String, GGMLQuantizationType>()
        val meta = loadFromStreamingGguf(ctx, dtype, { name, tensor -> byName[name] = tensor }) { name, qt ->
            quantTypes[name] = qt
        }
        return LlamaWeights(meta, byName, quantTypes)
    }

    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext
    ): LlamaWeights<T, V> = loadToMapStreaming(ctx, T::class)

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?
    ): LlamaModelMetadata {
        require(dtype == FP32::class || dtype == FP16::class) {
            "LLaMA GGUF loader supports FP32 and FP16 tensors (got ${dtype.simpleName})"
        }
        requireNotNull(sourceProvider) {
            "Sequential loading requires sourceProvider constructor. Use loadFromStreamingGguf for RandomAccessSource."
        }

        val reader = sourceProvider.invoke().buffered().use { src ->
            GGUFReader(src, loadTensorData = loadTensorData)
        }

        val metadata = metadataFromGguf(reader.fields, reader.tensors)
        validateMetadata(metadata)

        val required = requiredTensorNames(metadata)
        val tensorByName = reader.tensors.associateBy { it.name }

        required.forEach { name ->
            val rt = tensorByName[name]
                ?: error("Missing required tensor in GGUF payload: $name")
            validateTensorShape(name, rt, metadata)
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt)
            onTensorLoaded(name, tensor)
            if ((quantPolicy == QuantPolicy.RAW_BYTES || quantPolicy == QuantPolicy.NATIVE_OPTIMIZED) && rt.tensorType != GGMLQuantizationType.F32) {
                quantCallback?.invoke(name, rt.tensorType)
            }
        }

        // Optional tensors (e.g., precomputed RoPE tables) if present and float32
        listOf(
            LlamaTensorNames.ROPE_FREQS_REAL,
            LlamaTensorNames.ROPE_FREQS_IMAG
        ).forEach { name ->
            val rt = tensorByName[name]
            if (rt != null && rt.tensorType == GGMLQuantizationType.F32) {
                val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt)
                onTensorLoaded(name, tensor)
                // optional tensors are expected to be F32; quant types are ignored here
            }
        }

        return metadata
    }

    /**
     * Load using streaming API - only parses metadata into memory, loads tensors on-demand.
     * Suitable for models >2GB that exceed Java array limits.
     */
    private fun <T : DType, V> loadFromStreamingGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?
    ): LlamaModelMetadata {
        require(dtype == FP32::class || dtype == FP16::class) {
            "LLaMA GGUF loader supports FP32 and FP16 tensors (got ${dtype.simpleName})"
        }
        requireNotNull(randomAccessProvider) {
            "Streaming loading requires randomAccessProvider constructor. Use loadFromGguf for Source."
        }

        val source = randomAccessProvider.invoke()
        return StreamingGGUFReader.open(source).use { reader ->
            val metadata = metadataFromStreamingGguf(reader.fields, reader.tensors)
            validateMetadata(metadata)

            val required = requiredTensorNames(metadata)
            val tensorByName = reader.tensors.associateBy { it.name }

            required.forEach { name ->
                val st = tensorByName[name]
                    ?: error("Missing required tensor in GGUF payload: $name")
                validateStreamingTensorShape(name, st, metadata)
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st)
                onTensorLoaded(name, tensor)
                val isRawQuant = when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> st.tensorType != GGMLQuantizationType.F32
                    QuantPolicy.NATIVE_OPTIMIZED -> st.tensorType != GGMLQuantizationType.F32
                        && st.tensorType != GGMLQuantizationType.F16
                        && st.tensorType != GGMLQuantizationType.BF16
                    QuantPolicy.DEQUANTIZE_TO_FP32 -> false
                }
                if (isRawQuant) {
                    quantCallback?.invoke(name, st.tensorType)
                }
            }

            // Optional tensors (e.g., precomputed RoPE tables) if present and float32
            listOf(
                LlamaTensorNames.ROPE_FREQS_REAL,
                LlamaTensorNames.ROPE_FREQS_IMAG
            ).forEach { name ->
                val st = tensorByName[name]
                if (st != null && st.tensorType == GGMLQuantizationType.F32) {
                    val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st)
                    onTensorLoaded(name, tensor)
                }
            }

            metadata
        }
    }

    /**
     * Extract metadata from StreamingGGUFReader fields (which are direct values, not ReaderField).
     */
    private fun metadataFromStreamingGguf(
        fields: Map<String, Any?>,
        tensors: List<StreamingTensorInfo>
    ): LlamaModelMetadata {
        val arch = (fields["general.architecture"] as? String) ?: "unknown"

        val embeddingLength = fields["llama.embedding_length"]?.toIntValue()
            ?: inferEmbeddingFromStreamingTensor(tensors)
        val contextLength = fields["llama.context_length"]?.toIntValue() ?: 0
        val blockCount = fields["llama.block_count"]?.toIntValue() ?: 0
        val headCount = fields["llama.attention.head_count"]?.toIntValue() ?: 0
        val kvHeadCount = fields["llama.attention.head_count_kv"]?.toIntValue() ?: headCount
        val feedForwardLength = fields["llama.feed_forward_length"]?.toIntValue() ?: 0
        val ropeDim = fields["llama.rope.dimension_count"]?.toIntValue()
        val vocabSize = fields["llama.vocab_size"]?.toIntValue()
            ?: inferVocabFromStreamingTensor(tensors)

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

    private fun Any?.toIntValue(): Int? = when (this) {
        is Int -> this
        is UInt -> this.toInt()
        is Long -> this.toInt()
        is ULong -> this.toInt()
        is Short -> this.toInt()
        is UShort -> this.toInt()
        is Byte -> this.toInt()
        is UByte -> this.toInt()
        else -> null
    }

    private fun inferEmbeddingFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }

    private fun validateStreamingTensorShape(name: String, tensor: StreamingTensorInfo, metadata: LlamaModelMetadata) {
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

            LlamaTensorNames.ROPE_FREQS_REAL, LlamaTensorNames.ROPE_FREQS_IMAG -> {
                val headSize = metadata.embeddingLength / metadata.headCount
                require(dims.size == 2 && dims[0] == metadata.contextLength && dims[1] == headSize / 2) {
                    val expectedShape = "[${metadata.contextLength}, ${headSize / 2}]"
                    "Tensor $name must be [seqLen, headSize/2]=$expectedShape shaped; got $dims"
                }
            }

            else -> {
                when {
                    name.contains("attn_norm") || name.contains("ffn_norm") -> {
                        require(dims.size == 1 && dims[0] == metadata.embeddingLength) {
                            "Tensor $name must be [${metadata.embeddingLength}] shaped; got $dims"
                        }
                    }

                    name.contains("attn_q") || name.contains("attn_output") -> {
                        require(dims.size == 2 && dims.all { it == metadata.embeddingLength }) {
                            "Tensor $name must be [dim, dim]; got $dims"
                        }
                    }

                    name.contains("attn_k") || name.contains("attn_v") -> {
                        val headSize = metadata.embeddingLength / metadata.headCount
                        val kvDim = metadata.kvHeadCount * headSize
                        val expectedProduct = metadata.embeddingLength * kvDim
                        require(dims.size == 2 && dims.product() == expectedProduct) {
                            "Tensor $name must have product [dim=${metadata.embeddingLength}]*[kv_dim=$kvDim]=$expectedProduct; got $dims with product ${dims.product()}"
                        }
                    }

                    name.contains("ffn_gate") || name.contains("ffn_up") -> {
                        val expected = metadata.feedForwardLength * metadata.embeddingLength
                        require(dims.size == 2 && dims.product() == expected) {
                            "Tensor $name must have product $expected; got $dims"
                        }
                    }

                    name.contains("ffn_down") -> {
                        val expected = metadata.embeddingLength * metadata.feedForwardLength
                        require(dims.size == 2 && dims.product() == expected) {
                            "Tensor $name must have product $expected; got $dims"
                        }
                    }
                }
            }
        }
    }

    /**
     * Convert streaming tensor data to Tensor, with dequantization if configured.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> streamingTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingGGUFReader,
        st: StreamingTensorInfo
    ): Tensor<T, V> {
        val shape = Shape(*st.shape.map { it.toInt() }.toIntArray())
        val bytes = reader.loadTensorData(st)

        return when (st.tensorType) {
            GGMLQuantizationType.F32 -> {
                val floats = bytesToFloatArray(bytes)
                createTensor(ctx, dtype, shape, floats)
            }

            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "F16/BF16 tensor ${st.name} requires dtype Int8 with quantPolicy=RAW_BYTES"
                        }
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }

                    QuantPolicy.DEQUANTIZE_TO_FP32,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        require(dtype == FP32::class || dtype == FP16::class) {
                            "Dequantizing ${st.tensorType} requires dtype FP32 or FP16; got ${dtype.simpleName}"
                        }
                        val floats = when (st.tensorType) {
                            GGMLQuantizationType.F16 -> dequantF16FromBytes(bytes)
                            GGMLQuantizationType.BF16 -> dequantBF16FromBytes(bytes)
                            else -> error("Unreachable")
                        }
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            GGMLQuantizationType.I8,
            GGMLQuantizationType.I16,
            GGMLQuantizationType.I32 -> error("Native type ${st.tensorType} not yet supported")

            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q8_1,
            GGMLQuantizationType.Q2_K,
            GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_K,
            GGMLQuantizationType.IQ4_NL,
            GGMLQuantizationType.IQ4_XS,
            GGMLQuantizationType.TQ1_0,
            GGMLQuantizationType.TQ2_0 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "Quantized tensor ${st.name} requires dtype Int8 with quantPolicy=RAW_BYTES"
                        }
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }

                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        // Store raw quantized bytes; dtype can be FP32 (mixed mode).
                        // Streaming reader preserves logical shape, so use byte-level shape.
                        // The tensor is technically mistyped but works via erasure;
                        // matmul dispatch inspects the actual TensorData type at runtime.
                        val byteShape = Shape(bytes.size)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(byteShape, Int8::class, bytes) as Tensor<T, V>
                    }

                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        require(dtype == FP32::class || dtype == FP16::class) {
                            "Dequantizing ${st.tensorType} requires dtype FP32 or FP16; got ${dtype.simpleName}"
                        }
                        val floats = dequantFromBytes(bytes, st.tensorType, st.nElements.toInt())
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            GGMLQuantizationType.UNKNOWN -> {
                // Unknown quantization type - fall back to raw bytes
                println("WARNING: Tensor '${st.name}' has unknown quantization type (raw value: ${st.rawTypeValue}). Storing as raw bytes.")
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "Unknown tensor type (raw: ${st.rawTypeValue}) for '${st.name}' requires dtype Int8 with quantPolicy=RAW_BYTES"
                        }
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        // Cannot dequantize unknown type - fall back to raw bytes with warning
                        println("WARNING: Cannot dequantize unknown type (raw: ${st.rawTypeValue}) for '${st.name}'. Falling back to raw bytes.")
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                }
            }

            else -> {
                // Fallback for any other unhandled types (shouldn't normally reach here)
                println("WARNING: Unhandled tensor type ${st.tensorType} for '${st.name}'. Storing as raw bytes.")
                @Suppress("UNCHECKED_CAST")
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray = DequantOps.bytesToFloatArray(bytes)
    private fun dequantF16FromBytes(bytes: ByteArray): FloatArray = DequantOps.dequantF16FromBytes(bytes)
    private fun dequantBF16FromBytes(bytes: ByteArray): FloatArray = DequantOps.dequantBF16FromBytes(bytes)
    private fun dequantFromBytes(bytes: ByteArray, tensorType: GGMLQuantizationType, nElems: Int): FloatArray =
        DequantOps.dequantFromBytes(bytes, tensorType, nElems)

    private fun metadataFromGguf(
        fields: Map<String, ReaderField>,
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
                    "Tensor $name must be [${
                        metadata.embeddingLength
                    }] shaped; got $dims"
                }
            }

            LlamaTensorNames.ROPE_FREQS_REAL, LlamaTensorNames.ROPE_FREQS_IMAG -> {
                val headSize = metadata.embeddingLength / metadata.headCount
                require(dims.size == 2 && dims[0] == metadata.contextLength && dims[1] == headSize / 2) {
                    val expectedShape = "[${metadata.contextLength}, ${headSize / 2}]"
                    "Tensor $name must be [seqLen, headSize/2]=$expectedShape shaped; got $dims"
                }
            }

            else -> {
                when {
                    name.contains("attn_norm") || name.contains("ffn_norm") -> {
                        require(dims.size == 1 && dims[0] == metadata.embeddingLength) {
                            "Tensor $name must be [${metadata.embeddingLength}] shaped; got $dims"
                        }
                    }

                    name.contains("attn_q") || name.contains("attn_output") -> {
                        // Q and O projections are [dim, dim]
                        require(dims.size == 2 && dims.all { it == metadata.embeddingLength }) {
                            "Tensor $name must be [dim, dim]; got $dims"
                        }
                    }

                    name.contains("attn_k") || name.contains("attn_v") -> {
                        // K and V projections support GQA: stored as [dim, kv_dim] in GGUF
                        val headSize = metadata.embeddingLength / metadata.headCount
                        val kvDim = metadata.kvHeadCount * headSize
                        val expectedProduct = metadata.embeddingLength * kvDim
                        require(dims.size == 2 && dims.product() == expectedProduct) {
                            "Tensor $name must have product [dim=${metadata.embeddingLength}]*[kv_dim=$kvDim]=$expectedProduct; got $dims with product ${dims.product()}"
                        }
                    }

                    name.contains("ffn_gate") || name.contains("ffn_up") -> {
                        val expected = metadata.feedForwardLength * metadata.embeddingLength
                        require(dims.size == 2 && dims.product() == expected) {
                            "Tensor $name must have product $expected; got $dims"
                        }
                    }

                    name.contains("ffn_down") -> {
                        val expected = metadata.embeddingLength * metadata.feedForwardLength
                        require(dims.size == 2 && dims.product() == expected) {
                            "Tensor $name must have product $expected; got $dims"
                        }
                    }
                }
            }
        }
    }

    private fun ReaderField.scalarInt(): Int {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        val value = (part as List<*>).firstOrNull()
            ?: error("Empty data part for field $name")
        return when (value) {
            is Int -> value
            is UInt -> value.toInt()
            is Long -> value.toInt()
            is ULong -> value.toInt()
            is Short -> value.toInt()
            is UShort -> value.toInt()
            is Byte -> value.toInt()
            is UByte -> value.toInt()
            else -> error("Unsupported scalar type ${value::class} for field $name")
        }
    }

    private fun ReaderField.stringValue(): String {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        val bytes = (part as List<Any>).mapNotNull {
            when (it) {
                is UByte -> it.toByte()
                is Byte -> it
                else -> null
            }
        }
        return bytes.toByteArray().decodeToString()
    }

    private fun inferEmbeddingFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        // For most LLMs, embedding_length < vocab_size, so we take the min
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        // For most LLMs, vocab_size > embedding_length, so we take the max
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }

    private fun List<Int>.product(): Int = fold(1) { acc, v -> acc * v }

    /**
     * Create a tensor from float data, transposing 2D tensors from column-major to row-major.
     * GGUF stores 2D tensors in column-major order, so we transpose them at load time.
     * The dtype parameter determines the GPU storage format (FP32 or FP16).
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> createTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        originalShape: Shape,
        data: FloatArray
    ): Tensor<T, V> {
        return if (originalShape.rank == 2) {
            // Transpose 2D tensors from column-major to row-major
            val rows = originalShape[0]
            val cols = originalShape[1]
            val transposed = DequantOps.transposeColumnMajorToRowMajor(data, rows, cols)
            // Shape is now [cols, rows] after transpose
            val newShape = Shape(cols, rows)
            ctx.fromFloatArray<T, Float>(newShape, dtype, transposed) as Tensor<T, V>
        } else {
            ctx.fromFloatArray<T, Float>(originalShape, dtype, data) as Tensor<T, V>
        }
    }

    private fun <T : DType, V> readerTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        rt: ReaderTensor
    ): Tensor<T, V> {
        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())
        return when (rt.tensorType) {
            GGMLQuantizationType.F32 -> {
                @Suppress("UNCHECKED_CAST")
                val floats = (if (rt.data.isEmpty()) reader.materialize(rt) else rt.data) as List<Float>
                createTensor(ctx, dtype, shape, floats.toFloatArray())
            }

            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "F16/BF16 tensor ${rt.name} requires dtype Int8 with quantPolicy=RAW_BYTES; got ${dtype.simpleName}"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }

                    QuantPolicy.DEQUANTIZE_TO_FP32,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        require(dtype == FP32::class || dtype == FP16::class) {
                            "Dequantizing ${rt.tensorType} requires dtype FP32 or FP16; got ${dtype.simpleName}"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val floats = when (rt.tensorType) {
                            GGMLQuantizationType.F16 -> dequantF16(raw)
                            GGMLQuantizationType.BF16 -> dequantBF16(raw)
                            else -> error("Unsupported native type ${rt.tensorType}")
                        }
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            GGMLQuantizationType.I8,
            GGMLQuantizationType.I16,
            GGMLQuantizationType.I32 -> error("Native type ${rt.tensorType} not yet supported in LLaMA loader")

            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q8_1,
            GGMLQuantizationType.Q2_K,
            GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_K,
            GGMLQuantizationType.IQ4_NL,
            GGMLQuantizationType.IQ4_XS,
            GGMLQuantizationType.TQ1_0,
            GGMLQuantizationType.TQ2_0 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "Quantized tensor ${rt.name} requires dtype Int8 with quantPolicy=RAW_BYTES; got ${dtype.simpleName}"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }

                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        // Store raw quantized bytes; dtype can be FP32 (mixed mode).
                        // The tensor is technically mistyped but works via erasure;
                        // matmul dispatch inspects the actual TensorData type at runtime.
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }

                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        require(dtype == FP32::class || dtype == FP16::class) {
                            "Dequantizing ${rt.tensorType} requires dtype FP32 or FP16; got ${dtype.simpleName}"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val floats = when (rt.tensorType) {
                            GGMLQuantizationType.Q4_0 -> dequantQ4_0(raw, rt.nElements)
                            GGMLQuantizationType.Q4_1 -> dequantQ4_1(raw, rt.nElements)
                            GGMLQuantizationType.Q5_0 -> dequantQ5_0(raw, rt.nElements)
                            GGMLQuantizationType.Q5_1 -> dequantQ5_1(raw, rt.nElements)
                            GGMLQuantizationType.Q8_0 -> dequantQ8_0(raw, rt.nElements)
                            GGMLQuantizationType.Q8_1 -> dequantQ8_1(raw, rt.nElements)
                            GGMLQuantizationType.Q2_K -> dequantQ2K(raw, rt.nElements)
                            GGMLQuantizationType.Q3_K -> dequantQ3K(raw, rt.nElements)
                            GGMLQuantizationType.Q4_K -> dequantQ4K(raw, rt.nElements)
                            GGMLQuantizationType.Q5_K -> dequantQ5K(raw, rt.nElements)
                            GGMLQuantizationType.Q6_K -> dequantQ6K(raw, rt.nElements)
                            GGMLQuantizationType.Q8_K -> dequantQ8K(raw, rt.nElements)
                            GGMLQuantizationType.IQ4_NL -> dequantIQ4NL(raw, rt.nElements)
                            GGMLQuantizationType.IQ4_XS -> dequantIQ4XS(raw, rt.nElements)
                            GGMLQuantizationType.TQ1_0 -> dequantTQ1_0(raw, rt.nElements)
                            GGMLQuantizationType.TQ2_0 -> dequantTQ2_0(raw, rt.nElements)
                            else -> error("Dequantization for ${rt.tensorType} not implemented yet")
                        }
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            GGMLQuantizationType.UNKNOWN -> {
                // Unknown quantization type - fall back to raw bytes
                println("WARNING: Tensor '${rt.name}' has unknown quantization type (raw value: ${rt.rawTypeValue}). Storing as raw bytes.")
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "Unknown tensor type (raw: ${rt.rawTypeValue}) for '${rt.name}' requires dtype Int8 with quantPolicy=RAW_BYTES"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        // Cannot dequantize unknown type - fall back to raw bytes with warning
                        println("WARNING: Cannot dequantize unknown type (raw: ${rt.rawTypeValue}) for '${rt.name}'. Falling back to raw bytes.")
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                }
            }

            else -> {
                // Fallback for any other unhandled types (shouldn't normally reach here)
                println("WARNING: Unhandled tensor type ${rt.tensorType} for '${rt.name}'. Storing as raw bytes.")
                val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                @Suppress("UNCHECKED_CAST")
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }
}
