package sk.ainet.models.llama

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.models.llama.LlamaWeightMapper
import sk.ainet.models.llama.LlamaWeights
import sk.ainet.models.llama.LlamaTensorNames
import sk.ainet.io.model.DataType
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.io.safetensors.StreamingSafeTensorInfo
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.math.pow
import kotlin.reflect.KClass

/**
 * Loads LLaMA weights from HuggingFace SafeTensors format and maps them to
 * the canonical GGUF tensor naming used by [LlamaWeightMapper].
 *
 * Handles:
 * - HuggingFace → GGUF tensor name mapping
 * - Q4 + .qb companion tensor dequantization to FP32
 * - BF16/F16 dequantization to FP32
 * - Shape normalization ([1, dim] norms → [dim])
 * - Tied word embeddings (output.weight = token_embd.weight)
 */
public class LlamaSafeTensorsLoader<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val metadata: LlamaModelMetadata,
    private val tiedEmbeddings: Boolean = false
) {

    /**
     * Load weights from SafeTensors file and return [LlamaRuntimeWeights].
     */
    public fun load(randomAccessProvider: () -> RandomAccessSource): LlamaRuntimeWeights<T> {
        val tensors = mutableMapOf<String, Tensor<T, Float>>()

        StreamingSafeTensorsReader.open(randomAccessProvider()).use { reader ->
            // Index all tensors by name for companion lookup
            val tensorInfoMap = reader.tensors.associateBy { it.name }

            // Process non-qb tensors
            for (info in reader.tensors) {
                if (info.name.endsWith(".qb")) continue  // handled as companion

                val canonicalName = HfTensorNameMapper.toCanonical(info.name) ?: continue

                val tensor = when (info.dataType) {
                    DataType.QUANT4 -> {
                        // Q4 with companion .qb scales → dequant to FP32
                        val qbName = info.name + ".qb"
                        val qbInfo = tensorInfoMap[qbName]
                        val rawQ4 = reader.loadTensorData(info)
                        val qbBytes = qbInfo?.let { reader.loadTensorData(it) }
                        val targetShape = inferShape(canonicalName, info.shape)
                        val floats = dequantQ4(rawQ4, qbBytes, info.shape, qbInfo?.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    DataType.BFLOAT16 -> {
                        val bytes = reader.loadTensorData(info)
                        val floats = dequantBF16(bytes)
                        val targetShape = normalizeNormShape(info.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    DataType.FLOAT16 -> {
                        val bytes = reader.loadTensorData(info)
                        val floats = dequantF16(bytes)
                        val targetShape = normalizeNormShape(info.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    DataType.FLOAT32 -> {
                        val bytes = reader.loadTensorData(info)
                        val floats = bytesToFloatArray(bytes)
                        val targetShape = normalizeNormShape(info.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    else -> {
                        println("WARNING: Skipping tensor '${info.name}' with unsupported dtype ${info.dtype}")
                        continue
                    }
                }

                tensors[canonicalName] = tensor
                println("  Loaded: ${info.name} (${info.dtype} ${info.shape}) → $canonicalName ${tensor.shape}")
            }
        }

        // Handle tied embeddings: reuse token_embd as output.weight
        if (tiedEmbeddings && !tensors.containsKey(LlamaTensorNames.OUTPUT_WEIGHT)) {
            val embedding = tensors[LlamaTensorNames.TOKEN_EMBEDDINGS]
                ?: error("tie_word_embeddings=true but token embedding not found")
            tensors[LlamaTensorNames.OUTPUT_WEIGHT] = embedding
            println("  Tied: ${LlamaTensorNames.OUTPUT_WEIGHT} → ${LlamaTensorNames.TOKEN_EMBEDDINGS}")
        }

        val weights = LlamaWeights<T, Float>(
            metadata = metadata,
            tensors = tensors
        )
        return LlamaWeightMapper.map(weights)
    }

    /**
     * Infer the target shape for a tensor, normalizing norm shapes.
     * For Q4 tensors, the target shape is the logical shape (not the packed shape).
     */
    private fun inferShape(canonicalName: String, originalShape: List<Long>): Shape {
        // Q4 tensors already have logical shapes in the header
        return Shape(*originalShape.map { it.toInt() }.toIntArray())
    }

    /**
     * Normalize norm shapes: [1, dim] → [dim] for 1D norm weights.
     */
    private fun normalizeNormShape(shape: List<Long>): Shape {
        return if (shape.size == 2 && shape[0] == 1L) {
            Shape(shape[1].toInt())
        } else {
            Shape(*shape.map { it.toInt() }.toIntArray())
        }
    }

    // ========== Q4 Dequantization ==========

    /**
     * Dequantize Q4 (4-bit quantized) tensor using companion .qb scale tensor.
     *
     * Q4 packs two 4-bit values per byte. The .qb tensor provides per-group
     * scale factors in FP32. Each group of 32 elements shares one scale factor.
     */
    private fun dequantQ4(
        q4Bytes: ByteArray,
        qbBytes: ByteArray?,
        q4Shape: List<Long>,
        qbShape: List<Long>?
    ): FloatArray {
        val rows = q4Shape[0].toInt()
        val cols = q4Shape[1].toInt()
        val totalElements = rows * cols
        val out = FloatArray(totalElements)

        if (qbBytes == null) {
            // No scale tensor; just unpack nibbles to [-8, 7] range
            for (i in 0 until totalElements) {
                val byteIdx = i / 2
                val nibble = if (i % 2 == 0) {
                    (q4Bytes[byteIdx].toInt() and 0x0F)
                } else {
                    (q4Bytes[byteIdx].toInt() shr 4) and 0x0F
                }
                // Signed 4-bit: 0..7 → 0..7, 8..15 → -8..-1
                out[i] = (if (nibble >= 8) nibble - 16 else nibble).toFloat()
            }
            return out
        }

        // Unpack .qb scales from FP32 bytes
        val scales = bytesToFloatArray(qbBytes)

        // Determine group size from shapes: each scale covers a group of elements
        // qb shape is [rows, numGroups] where numGroups = cols / groupSize
        val numGroups = if (qbShape != null && qbShape.size == 2) qbShape[1].toInt() else scales.size / rows
        val groupSize = if (numGroups > 0) cols / numGroups else 32

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val flatIdx = row * cols + col
                val byteIdx = flatIdx / 2
                val nibble = if (flatIdx % 2 == 0) {
                    (q4Bytes[byteIdx].toInt() and 0x0F)
                } else {
                    (q4Bytes[byteIdx].toInt() shr 4) and 0x0F
                }
                // Signed 4-bit
                val q = if (nibble >= 8) nibble - 16 else nibble
                // Look up group scale
                val groupIdx = col / groupSize
                val scaleIdx = row * numGroups + groupIdx
                val scale = if (scaleIdx < scales.size) scales[scaleIdx] else 1.0f
                out[flatIdx] = q.toFloat() * scale
            }
        }

        return out
    }

    // ========== BF16/F16 Dequantization ==========

    private fun dequantBF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val bf16Low = bytes[offset].toInt() and 0xFF
            val bf16High = bytes[offset + 1].toInt() and 0xFF
            val bits = (bf16High shl 24) or (bf16Low shl 16)
            out[i] = Float.fromBits(bits)
        }
        return out
    }

    private fun dequantF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val half = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8)
            out[i] = halfToFloat(half)
        }
        return out
    }

    private fun halfToFloat(hbits: Int): Float {
        val mant = hbits and 0x03FF
        val exp = hbits and 0x7C00
        val sign = hbits and 0x8000
        return when (exp) {
            0 -> {
                val v = (mant.toFloat() / 1024.0f) * (2.0f).pow(-14)
                if (sign != 0) -v else v
            }
            0x7C00 -> {
                val v = if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                if (sign != 0) -v else v
            }
            else -> {
                val v = (1.0f + mant.toFloat() / 1024.0f) * (2.0f).pow((exp shr 10) - 15)
                if (sign != 0) -v else v
            }
        }
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            val offset = i * 4
            val bits = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
            out[i] = Float.fromBits(bits)
        }
        return out
    }
}

/**
 * Maps HuggingFace tensor names to GGUF canonical names used by [LlamaTensorNames].
 */
public object HfTensorNameMapper {

    private val LAYER_PATTERN = Regex("""model\.layers\.(\d+)\.(.+)""")

    /**
     * Convert a HuggingFace tensor name to its GGUF canonical equivalent.
     * Returns null if the tensor should be skipped (e.g. .qb companions).
     */
    public fun toCanonical(hfName: String): String? {
        // Global tensors
        return when (hfName) {
            "model.embed_tokens.weight" -> LlamaTensorNames.TOKEN_EMBEDDINGS
            "model.norm.weight" -> LlamaTensorNames.OUTPUT_NORM
            "lm_head.weight" -> LlamaTensorNames.OUTPUT_WEIGHT
            else -> {
                // Layer tensors
                val match = LAYER_PATTERN.matchEntire(hfName) ?: return null
                val layer = match.groupValues[1].toInt()
                when (match.groupValues[2]) {
                    "input_layernorm.weight" -> LlamaTensorNames.attnNorm(layer)
                    "self_attn.q_proj.weight" -> LlamaTensorNames.attnQ(layer)
                    "self_attn.k_proj.weight" -> LlamaTensorNames.attnK(layer)
                    "self_attn.v_proj.weight" -> LlamaTensorNames.attnV(layer)
                    "self_attn.o_proj.weight" -> LlamaTensorNames.attnOut(layer)
                    "post_attention_layernorm.weight" -> LlamaTensorNames.ffnNorm(layer)
                    "mlp.gate_proj.weight" -> LlamaTensorNames.ffnGate(layer)
                    "mlp.down_proj.weight" -> LlamaTensorNames.ffnDown(layer)
                    "mlp.up_proj.weight" -> LlamaTensorNames.ffnUp(layer)
                    else -> null
                }
            }
        }
    }
}
