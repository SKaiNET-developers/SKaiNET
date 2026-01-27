package sk.ainet.io.gguf.llama

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
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
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

    public enum class QuantPolicy {
        /** Keep quantized payloads as raw bytes (Int8 tensor) with quantized shape. */
        RAW_BYTES,

        /**
         * Dequantize to FP32 on load. Currently unsupported; use RAW_BYTES until a dequant path
         * is implemented.
         */
        DEQUANTIZE_TO_FP32
    }

    public companion object Dequant {
        private fun typeName(value: Any?): String =
            value?.let { it::class.simpleName ?: it::class.toString() } ?: "null"

        /**
         * Handle column-major to row-major conversion for GGUF tensors.
         *
         * GGUF stores 2D tensors in column-major order with shape [in_dim, out_dim].
         * For a weight matrix W where y = x @ W:
         * - W[i, j] represents weight from input dimension i to output dimension j
         * - In column-major, W[i, j] is at data[i + j * in_dim]
         *
         * After swapping shape to [out_dim, in_dim] and interpreting as row-major:
         * - Element at row j, col i is at data[j * in_dim + i]
         * - This equals i + j * in_dim (addition is commutative)
         * - So we access W[i, j] as intended
         *
         * The data doesn't need to change - only the shape interpretation changes.
         */
        @Suppress("UNUSED_PARAMETER")
        internal fun transposeColumnMajorToRowMajor(
            data: FloatArray,
            rows: Int,
            cols: Int
        ): FloatArray {
            // Data layout is unchanged - we just swap the shape dimensions
            return data
        }

        @OptIn(ExperimentalUnsignedTypes::class)
        private fun toByteArray(raw: List<Any>, tensorName: String): ByteArray {
            val first = raw.firstOrNull()
            return when (first) {
                is Byte -> ByteArray(raw.size) { (raw[it] as Number).toByte() }
                is UByte -> ByteArray(raw.size) { (raw[it] as UByte).toByte() }
                else -> error("Unexpected raw data type ${typeName(first)} for tensor $tensorName")
            }
        }

        internal fun dequantF16(raw: List<Any>): FloatArray {
            val bytes: ByteArray = toByteArray(raw, "F16")
            val out = FloatArray(bytes.size / 2)
            var i = 0
            var o = 0
            while (i < bytes.size) {
                val b0 = bytes[i].toInt() and 0xFF
                val b1 = bytes[i + 1].toInt() and 0xFF
                val half = (b1 shl 8) or b0
                out[o] = halfToFloat(half)
                i += 2
                o++
            }
            return out
        }

        internal fun dequantBF16(raw: List<Any>): FloatArray {
            val bytes: ByteArray = toByteArray(raw, "BF16")
            val out = FloatArray(bytes.size / 2)
            var i = 0
            var o = 0
            while (i < bytes.size) {
                val b0 = bytes[i].toInt() and 0xFF
                val b1 = bytes[i + 1].toInt() and 0xFF
                // BF16 stores exponent and mantissa in upper 16 bits of IEEE754 float
                val bits = (b1 shl 24) or (b0 shl 16)
                out[o] = Float.fromBits(bits)
                i += 2
                o++
            }
            return out
        }

        private fun halfToFloat(hbits: Int): Float {
            val mant = hbits and 0x03FF
            val exp = hbits and 0x7C00
            val sign = hbits and 0x8000
            return when (exp) {
                0 -> {
                    // subnormal
                    val v = (mant.toFloat() / 1024.0f) * (2.0f).pow(-14)
                    if (sign != 0) -v else v
                }

                0x7C00 -> {
                    // Inf/NaN
                    val v = if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                    if (sign != 0) -v else v
                }

                else -> {
                    val v = (1.0f + mant.toFloat() / 1024.0f) * (2.0f).pow((exp shr 10) - 15)
                    if (sign != 0) -v else v
                }
            }
        }

        internal fun dequantQ4_0(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q4_0")
            val blockSize = 32
            val bytesPerBlock = 18 // 2 (f16 scale) + 16 (32 nibbles)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                // Using unsigned conversion for the bytes before assembling the 16-bit value
                val b0 = bytes[offset].toInt() and 0xFF
                val b1 = bytes[offset + 1].toInt() and 0xFF
                val d = halfToFloat(b0 or (b1 shl 8))
                offset += 2
                for (j in 0 until 16) {
                    val b = bytes[offset + j].toInt() and 0xFF
                    val lo = (b and 0x0F) - 8
                    val hi = (b shr 4) - 8
                    out[outOff + j] = lo.toFloat() * d
                    out[outOff + 16 + j] = hi.toFloat() * d
                }
                offset += 16
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ5_0(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q5_0")
            val blockSize = 32
            val bytesPerBlock = 22 // 2 (f16 scale) + 4 (qh) + 16 (qs)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
                offset += 2
                val qh0 = bytes[offset].toInt() and 0xFF
                val qh1 = bytes[offset + 1].toInt() and 0xFF
                val qh2 = bytes[offset + 2].toInt() and 0xFF
                val qh3 = bytes[offset + 3].toInt() and 0xFF
                offset += 4
                val qh = intArrayOf(qh0, qh1, qh2, qh3)
                for (j in 0 until 16) {
                    val q = bytes[offset + j].toInt() and 0xFF
                    val lo = q and 0x0F
                    val hi = q shr 4
                    val bitLo = ((qh[j / 8] shr (j % 8)) and 0x01) shl 4
                    val bitHi = ((qh[(j + 16) / 8] shr ((j + 16) % 8)) and 0x01) shl 4
                    out[outOff + j] = d * (lo + bitLo - 16)
                    out[outOff + 16 + j] = d * (hi + bitHi - 16)
                }
                offset += 16
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ8_0(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q8_0")
            val blockSize = 32
            val bytesPerBlock = 34 // 2 (f16 scale) + 32 (qs)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
                offset += 2
                for (j in 0 until 32) {
                    out[outOff + j] = d * bytes[offset + j].toFloat()
                }
                offset += 32
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ4_1(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q4_1")
            val blockSize = 32
            val bytesPerBlock = 20 // 2 (f16 d) + 2 (f16 m) + 16 (qs)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
                val m = halfToFloat((bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF))
                offset += 4
                for (j in 0 until 16) {
                    val b = bytes[offset + j].toInt() and 0xFF
                    val lo = b and 0x0F
                    val hi = b shr 4
                    out[outOff + j] = d * lo + m
                    out[outOff + 16 + j] = d * hi + m
                }
                offset += 16
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ5_1(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q5_1")
            val blockSize = 32
            val bytesPerBlock = 24 // 2 (f16 d) + 2 (f16 m) + 4 (qh) + 16 (qs)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
                val m = halfToFloat((bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF))
                offset += 4
                val qh0 = bytes[offset].toInt() and 0xFF
                val qh1 = bytes[offset + 1].toInt() and 0xFF
                val qh2 = bytes[offset + 2].toInt() and 0xFF
                val qh3 = bytes[offset + 3].toInt() and 0xFF
                offset += 4
                val qh = intArrayOf(qh0, qh1, qh2, qh3)
                for (j in 0 until 16) {
                    val q = bytes[offset + j].toInt() and 0xFF
                    val lo = q and 0x0F
                    val hi = q shr 4
                    val bitLo = (qh[j / 8] shr (j % 8)) and 0x01
                    val bitHi = (qh[(j + 16) / 8] shr ((j + 16) % 8)) and 0x01
                    out[outOff + j] = d * (lo + (bitLo shl 4)) + m
                    out[outOff + 16 + j] = d * (hi + (bitHi shl 4)) + m
                }
                offset += 16
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ8_1(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q8_1")
            val blockSize = 32
            val bytesPerBlock = 40 // 4 (f32 d) + 4 (f32 s) + 32 (qs)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                // Read f32 d (little-endian)
                val dBits = (bytes[offset].toInt() and 0xFF) or
                        ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                        ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                        ((bytes[offset + 3].toInt() and 0xFF) shl 24)
                val d = Float.fromBits(dBits)
                // Read f32 s (little-endian) - this is d * sum(qs), used as min offset
                val sBits = (bytes[offset + 4].toInt() and 0xFF) or
                        ((bytes[offset + 5].toInt() and 0xFF) shl 8) or
                        ((bytes[offset + 6].toInt() and 0xFF) shl 16) or
                        ((bytes[offset + 7].toInt() and 0xFF) shl 24)
                val s = Float.fromBits(sBits)
                offset += 8
                for (j in 0 until 32) {
                    out[outOff + j] = d * bytes[offset + j].toFloat()
                }
                offset += 32
                outOff += blockSize
            }
            return out
        }

        private val iq4nlValues: IntArray = intArrayOf(
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
        )

        internal fun dequantIQ4NL(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "IQ4_NL")
            val blockSize = 32
            val bytesPerBlock = 18 // 2 (f16 d) + 16 (qs)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
                offset += 2
                repeat(blockSize / 2) { j ->
                    val code = bytes[offset + j].toInt() and 0xFF
                    val lo = code and 0x0F
                    val hi = code ushr 4
                    out[outOff + j] = d * iq4nlValues[lo]
                    out[outOff + blockSize / 2 + j] = d * iq4nlValues[hi]
                }
                offset += blockSize / 2
                outOff += blockSize
            }
            return out
        }

        internal fun dequantIQ4XS(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "IQ4_XS")
            val blockSize = QK_K
            val bytesPerBlock = 2 + 2 + QK_K / 2 + QK_K / 64 // d + scalesH + qs + scalesL = 136
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
                offset += 2
                val scalesH = (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                offset += 2
                val scalesL = bytes.copyOfRange(offset, offset + QK_K / 64)
                offset += QK_K / 64
                val qs = bytes.copyOfRange(offset, offset + QK_K / 2)
                offset += QK_K / 2
                repeat(QK_K / 32) { ib ->
                    val ls = ((scalesL[ib / 2].toInt() ushr (4 * (ib % 2))) and 0x0F) or
                        (((scalesH ushr (2 * ib)) and 0x03) shl 4)
                    val dl = d * (ls - 32)
                    repeat(16) { j ->
                        val code = qs[ib * 16 + j].toInt() and 0xFF
                        val lo = code and 0x0F
                        val hi = code ushr 4
                        out[outOff + ib * 32 + j] = dl * iq4nlValues[lo]
                        out[outOff + ib * 32 + 16 + j] = dl * iq4nlValues[hi]
                    }
                }
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ2K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q2_K")
            val blockSize = QK_K
            val bytesPerBlock = 2 + 2 + QK_K / 16 + QK_K / 4 // d + dMin + scales + qs = 84
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat(
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                )
                val dMin = halfToFloat(
                    (bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF)
                )
                offset += 4
                val scales = bytes.copyOfRange(offset, offset + 16)
                offset += 16
                val qs = bytes.copyOfRange(offset, offset + 64)
                offset += 64
                repeat(16) { block ->
                    val scaleIdx = (scales[block].toInt() ushr 4) and 0x0F
                    val minIdx = scales[block].toInt() and 0x0F
                    val scale = d * (scaleIdx / 15.0f)
                    // Q2_K formula: y = d * q - dmin * m (subtraction, not addition)
                    val min = dMin * (minIdx / 15.0f)
                    repeat(16) { j ->
                        val codeByte = qs[block * 4 + j / 4].toInt() and 0xFF
                        val q = (codeByte ushr ((j % 4) * 2)) and 0x03
                        out[outOff + block * 16 + j] = q * scale - min
                    }
                }
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ3K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q3_K")
            val blockSize = QK_K
            val bytesPerBlock = 2 + QK_K / 4 + QK_K / 8 + 12 // d + hmask + qs + scales = 110
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat(
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                )
                offset += 2
                val hmask = bytes.copyOfRange(offset, offset + 32)
                offset += 32
                val qs = bytes.copyOfRange(offset, offset + 64)
                offset += 64
                val scales = bytes.copyOfRange(offset, offset + 12)
                offset += 12
                repeat(16) { block ->
                    val bitPos = block * 6
                    val bytePos = bitPos / 8
                    val bitShift = bitPos % 8
                    val packed =
                        (scales.getOrElse(bytePos) { 0 }.toInt() and 0xFF) or
                            ((scales.getOrElse(bytePos + 1) { 0 }.toInt() and 0xFF) shl 8) or
                            ((scales.getOrElse(bytePos + 2) { 0 }.toInt() and 0xFF) shl 16)
                    val scaleIdx = (packed ushr bitShift) and 0x3F
                    val scale = d * (scaleIdx / 63.0f)
                    repeat(16) { j ->
                        val idx = block * 16 + j
                        val ql = (qs[idx / 4].toInt() ushr ((idx % 4) * 2)) and 0x03
                        val qh = (hmask[idx / 8].toInt() ushr (idx % 8)) and 0x01
                        val q = ql or (qh shl 2)
                        out[outOff + idx] = q * scale
                    }
                }
                outOff += blockSize
            }
            return out
        }

        /**
         * Helper to extract scale and min indices for Q4_K and Q5_K formats.
         * Matches llama.cpp's get_scale_min_k4() function.
         */
        private fun getScaleMinK4(j: Int, scales: ByteArray): Pair<Int, Int> {
            return if (j < 4) {
                val sc = scales[j].toInt() and 0x3F
                val m = scales[j + 4].toInt() and 0x3F
                sc to m
            } else {
                val sc = ((scales[j + 4].toInt() and 0x0F) or
                         (((scales[j - 4].toInt() and 0xFF) shr 6) shl 4))
                val m = (((scales[j + 4].toInt() and 0xFF) shr 4) or
                        (((scales[j].toInt() and 0xFF) shr 6) shl 4))
                sc to m
            }
        }

        internal fun dequantQ4K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q4_K")
            val blockSize = QK_K
            val bytesPerBlock = 144 // 2 (f16 d) + 2 (f16 dMin) + 12 (scales) + 128 (qs)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat(
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                )
                val dMin = halfToFloat(
                    (bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF)
                )
                offset += 4
                val scales = bytes.copyOfRange(offset, offset + 12)
                offset += 12
                val qs = bytes.copyOfRange(offset, offset + 128)
                offset += 128

                // Process 256 elements in groups of 64 (matching llama.cpp structure)
                var qOffset = 0
                var scaleIdx = 0
                repeat(4) { group ->
                    // Get scales for this pair of 32-element sub-blocks
                    val (sc1, m1) = getScaleMinK4(scaleIdx, scales)
                    val (sc2, m2) = getScaleMinK4(scaleIdx + 1, scales)
                    val d1 = d * sc1
                    val min1 = dMin * m1
                    val d2 = d * sc2
                    val min2 = dMin * m2

                    // First 32 elements: lower nibbles
                    for (l in 0 until 32) {
                        val q = qs[qOffset + l].toInt() and 0x0F
                        out[outOff++] = d1 * q - min1
                    }
                    // Next 32 elements: upper nibbles
                    for (l in 0 until 32) {
                        val q = (qs[qOffset + l].toInt() and 0xFF) shr 4
                        out[outOff++] = d2 * q - min2
                    }
                    qOffset += 32
                    scaleIdx += 2
                }
            }
            return out
        }

        internal fun dequantQ5K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q5_K")
            val blockSize = QK_K
            val bytesPerBlock = 176 // 2 (f16 d) + 2 (f16 dMin) + 12 (scales) + 32 (qh) + 128 (qs)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat(
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                )
                val dMin = halfToFloat(
                    (bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF)
                )
                offset += 4
                val scales = bytes.copyOfRange(offset, offset + 12)
                offset += 12
                val qh = bytes.copyOfRange(offset, offset + 32)
                offset += 32
                val qs = bytes.copyOfRange(offset, offset + 128)
                offset += 128

                // Process 256 elements in groups of 64 (matching llama.cpp structure)
                var qOffset = 0
                var scaleIdx = 0
                var outIdx = 0
                repeat(4) { group ->
                    val (sc1, m1) = getScaleMinK4(scaleIdx, scales)
                    val (sc2, m2) = getScaleMinK4(scaleIdx + 1, scales)
                    val d1 = d * sc1
                    val min1 = dMin * m1
                    val d2 = d * sc2
                    val min2 = dMin * m2

                    // First 32 elements: lower nibbles with high bit from qh
                    for (l in 0 until 32) {
                        val idx = outIdx + l
                        val qLow = qs[qOffset + l].toInt() and 0x0F
                        val qHigh = ((qh[idx / 8].toInt() and 0xFF) shr (idx % 8)) and 0x01
                        val q = qLow or (qHigh shl 4)
                        out[outOff + idx] = d1 * q - min1
                    }
                    // Next 32 elements: upper nibbles with high bit from qh
                    for (l in 0 until 32) {
                        val idx = outIdx + 32 + l
                        val qLow = ((qs[qOffset + l].toInt() and 0xFF) shr 4)
                        val qHigh = ((qh[idx / 8].toInt() and 0xFF) shr (idx % 8)) and 0x01
                        val q = qLow or (qHigh shl 4)
                        out[outOff + idx] = d2 * q - min2
                    }
                    qOffset += 32
                    scaleIdx += 2
                    outIdx += 64
                }
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ6K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q6_K")
            val blockSize = QK_K
            // Q6_K block layout: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes
            val bytesPerBlock = 210
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                // Read ql (128 bytes) - lower 4 bits of each 6-bit value
                val ql = bytes.copyOfRange(offset, offset + 128)
                offset += 128
                // Read qh (64 bytes) - upper 2 bits of each 6-bit value
                val qh = bytes.copyOfRange(offset, offset + 64)
                offset += 64
                // Read scales (16 signed int8 values)
                val scales = bytes.copyOfRange(offset, offset + 16)
                offset += 16
                // Read d (f16 scale factor)
                val d = halfToFloat(
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                )
                offset += 2

                // Process two 128-element halves (matching llama.cpp layout)
                // Each half uses 64 bytes of ql, 32 bytes of qh, and 8 scales
                repeat(2) { half ->
                    val qlBase = half * 64
                    val qhBase = half * 32
                    val scBase = half * 8

                    for (l in 0 until 32) {
                        val isIdx = l / 16  // 0 for l < 16, 1 for l >= 16

                        // q1: lower nibble of ql[l], qh bits 0-1
                        val q1Low = ql[qlBase + l].toInt() and 0x0F
                        val q1High = (qh[qhBase + l].toInt() shr 0) and 0x03
                        val q1 = (q1Low or (q1High shl 4)) - 32

                        // q2: lower nibble of ql[l+32], qh bits 2-3
                        val q2Low = ql[qlBase + l + 32].toInt() and 0x0F
                        val q2High = (qh[qhBase + l].toInt() shr 2) and 0x03
                        val q2 = (q2Low or (q2High shl 4)) - 32

                        // q3: upper nibble of ql[l], qh bits 4-5
                        val q3Low = (ql[qlBase + l].toInt() and 0xFF) shr 4
                        val q3High = (qh[qhBase + l].toInt() shr 4) and 0x03
                        val q3 = (q3Low or (q3High shl 4)) - 32

                        // q4: upper nibble of ql[l+32], qh bits 6-7
                        val q4Low = (ql[qlBase + l + 32].toInt() and 0xFF) shr 4
                        val q4High = (qh[qhBase + l].toInt() shr 6) and 0x03
                        val q4 = (q4Low or (q4High shl 4)) - 32

                        // Scale indices: isIdx+0, isIdx+2, isIdx+4, isIdx+6
                        val sc1 = scales[scBase + isIdx + 0].toInt()
                        val sc2 = scales[scBase + isIdx + 2].toInt()
                        val sc3 = scales[scBase + isIdx + 4].toInt()
                        val sc4 = scales[scBase + isIdx + 6].toInt()

                        out[outOff + half * 128 + l + 0] = d * sc1 * q1
                        out[outOff + half * 128 + l + 32] = d * sc2 * q2
                        out[outOff + half * 128 + l + 64] = d * sc3 * q3
                        out[outOff + half * 128 + l + 96] = d * sc4 * q4
                    }
                }
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ8K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q8_K")
            val blockSize = QK_K
            val bytesPerBlock = 292 // 4 (f32 d) + 256 (qs) + 32 (bsums)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val dBits =
                    (bytes[offset + 3].toInt() and 0xFF shl 24) or
                        (bytes[offset + 2].toInt() and 0xFF shl 16) or
                        (bytes[offset + 1].toInt() and 0xFF shl 8) or
                        (bytes[offset].toInt() and 0xFF)
                val d = Float.fromBits(dBits)
                offset += 4
                repeat(blockSize) { j ->
                    out[outOff + j] = d * bytes[offset + j].toFloat()
                }
                offset += blockSize
                // Skip bsums (16 * int16) even though they are not needed for dequant
                offset += 32
                outOff += blockSize
            }
            return out
        }

        /**
         * Dequantize TQ2_0 (Ternary 2-bit) format to FP32.
         *
         * TQ2_0 layout per block (256 elements, 66 bytes):
         * - 64 bytes: quantized data (4 ternary values per byte, 2-bit each)
         * - 2 bytes: f16 scale
         *
         * Values encoded as {0, 1, 2} represent {-1, 0, +1}.
         * Dequantization: output[i] = (ternary[i] - 1) * scale
         */
        internal fun dequantTQ2_0(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "TQ2_0")
            val blockSize = 256
            val bytesPerBlock = 66 // 64 (qs) + 2 (f16 scale)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0

            repeat(blockCount) {
                // Read quantized values first (64 bytes = 256 values at 2-bit each)
                val qs = bytes.copyOfRange(offset, offset + 64)
                offset += 64

                // Read f16 scale (last 2 bytes)
                val scale = halfToFloat(
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                )
                offset += 2

                // Decode 2-bit values: 4 values per byte
                // Bit layout: [v3:v2:v1:v0] where each vN is 2 bits
                for (i in 0 until 64) {
                    val b = qs[i].toInt() and 0xFF
                    val v0 = (b and 0x03) - 1         // bits 0-1
                    val v1 = ((b shr 2) and 0x03) - 1 // bits 2-3
                    val v2 = ((b shr 4) and 0x03) - 1 // bits 4-5
                    val v3 = ((b shr 6) and 0x03) - 1 // bits 6-7

                    out[outOff + i * 4 + 0] = v0 * scale
                    out[outOff + i * 4 + 1] = v1 * scale
                    out[outOff + i * 4 + 2] = v2 * scale
                    out[outOff + i * 4 + 3] = v3 * scale
                }
                outOff += blockSize
            }
            return out
        }

        /**
         * Dequantize TQ1_0 (Ternary base-3) format to FP32.
         *
         * TQ1_0 layout per block (256 elements, 54 bytes):
         * - 48 bytes: base-3 packed data (5 values per byte, 240 elements total)
         * - 4 bytes: 2-bit packed for remaining 16 elements
         * - 2 bytes: f16 scale
         *
         * Base-3 encoding: 5 ternary values packed into one byte (3^5 = 243 < 256).
         * Values {0, 1, 2} represent {-1, 0, +1}.
         * Dequantization: output[i] = (ternary[i] - 1) * scale
         */
        internal fun dequantTQ1_0(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "TQ1_0")
            val blockSize = 256
            val bytesPerBlock = 54 // 48 (base-3) + 4 (2-bit) + 2 (f16 scale)
            val blockCount = bytes.size / bytesPerBlock
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0

            repeat(blockCount) {
                // Read base-3 packed data (48 bytes = 240 elements)
                val qsBase3 = bytes.copyOfRange(offset, offset + 48)
                offset += 48

                // Read 2-bit packed data for remaining 16 elements (4 bytes)
                val qs2bit = bytes.copyOfRange(offset, offset + 4)
                offset += 4

                // Read f16 scale
                val scale = halfToFloat(
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                )
                offset += 2

                // Decode base-3 packed values (5 values per byte)
                // Each byte b encodes: v0 + v1*3 + v2*9 + v3*27 + v4*81
                var outIdx = 0
                for (i in 0 until 48) {
                    var b = qsBase3[i].toInt() and 0xFF
                    repeat(5) {
                        val v = (b % 3) - 1  // Extract value and convert to {-1, 0, +1}
                        out[outOff + outIdx] = v * scale
                        outIdx++
                        b /= 3
                    }
                }

                // Decode remaining 16 elements from 2-bit packing (4 bytes)
                for (i in 0 until 4) {
                    val b = qs2bit[i].toInt() and 0xFF
                    val v0 = (b and 0x03) - 1
                    val v1 = ((b shr 2) and 0x03) - 1
                    val v2 = ((b shr 4) and 0x03) - 1
                    val v3 = ((b shr 6) and 0x03) - 1

                    out[outOff + 240 + i * 4 + 0] = v0 * scale
                    out[outOff + 240 + i * 4 + 1] = v1 * scale
                    out[outOff + 240 + i * 4 + 2] = v2 * scale
                    out[outOff + 240 + i * 4 + 3] = v3 * scale
                }

                outOff += blockSize
            }
            return out
        }
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
        require(dtype == FP32::class) {
            "LLaMA GGUF loader currently supports FP32 tensors only (got ${dtype.simpleName})"
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
            if (quantPolicy == QuantPolicy.RAW_BYTES && rt.tensorType != GGMLQuantizationType.F32) {
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
        require(dtype == FP32::class) {
            "LLaMA GGUF loader currently supports FP32 tensors only (got ${dtype.simpleName})"
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
                if (quantPolicy == QuantPolicy.RAW_BYTES && st.tensorType != GGMLQuantizationType.F32) {
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
                createFp32Tensor(ctx, dtype, shape, floats)
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

                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        require(dtype == FP32::class) {
                            "Dequantizing ${st.tensorType} to FP32 requires dtype FP32"
                        }
                        val floats = when (st.tensorType) {
                            GGMLQuantizationType.F16 -> dequantF16FromBytes(bytes)
                            GGMLQuantizationType.BF16 -> dequantBF16FromBytes(bytes)
                            else -> error("Unreachable")
                        }
                        createFp32Tensor(ctx, dtype, shape, floats)
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

                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        require(dtype == FP32::class) {
                            "Dequantizing ${st.tensorType} to FP32 requires dtype FP32"
                        }
                        val floats = dequantFromBytes(bytes, st.tensorType, st.nElements.toInt())
                        createFp32Tensor(ctx, dtype, shape, floats)
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
                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        // Cannot dequantize unknown type - fall back to raw bytes with warning
                        println("WARNING: Cannot dequantize unknown type (raw: ${st.rawTypeValue}) for '${st.name}'. Falling back to raw bytes.")
                        require(dtype == Int8::class) {
                            "Unknown tensor type cannot be dequantized. Use dtype Int8 or add support for type ${st.rawTypeValue}"
                        }
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                }
            }

            else -> {
                // Fallback for any other unhandled types (shouldn't normally reach here)
                println("WARNING: Unhandled tensor type ${st.tensorType} for '${st.name}'. Storing as raw bytes.")
                require(dtype == Int8::class) {
                    "Unhandled tensor type ${st.tensorType} requires dtype Int8"
                }
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }

    /**
     * Convert raw bytes (little-endian) to float array.
     */
    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val bits = (bytes[i].toInt() and 0xFF) or
                ((bytes[i + 1].toInt() and 0xFF) shl 8) or
                ((bytes[i + 2].toInt() and 0xFF) shl 16) or
                ((bytes[i + 3].toInt() and 0xFF) shl 24)
            out[o] = Float.fromBits(bits)
            i += 4
            o++
        }
        return out
    }

    /**
     * Dequantize F16 bytes to float array.
     */
    private fun dequantF16FromBytes(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val b0 = bytes[i].toInt() and 0xFF
            val b1 = bytes[i + 1].toInt() and 0xFF
            val half = (b1 shl 8) or b0
            out[o] = halfToFloat(half)
            i += 2
            o++
        }
        return out
    }

    /**
     * Dequantize BF16 bytes to float array.
     */
    private fun dequantBF16FromBytes(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val b0 = bytes[i].toInt() and 0xFF
            val b1 = bytes[i + 1].toInt() and 0xFF
            val bits = (b1 shl 24) or (b0 shl 16)
            out[o] = Float.fromBits(bits)
            i += 2
            o++
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

    /**
     * Dispatch dequantization based on tensor type for byte arrays.
     */
    private fun dequantFromBytes(bytes: ByteArray, tensorType: GGMLQuantizationType, nElems: Int): FloatArray {
        // Convert bytes to List<Any> to reuse existing dequant functions
        val raw: List<Any> = bytes.map { it }
        return when (tensorType) {
            GGMLQuantizationType.Q4_0 -> dequantQ4_0(raw, nElems)
            GGMLQuantizationType.Q4_1 -> dequantQ4_1(raw, nElems)
            GGMLQuantizationType.Q5_0 -> dequantQ5_0(raw, nElems)
            GGMLQuantizationType.Q5_1 -> dequantQ5_1(raw, nElems)
            GGMLQuantizationType.Q8_0 -> dequantQ8_0(raw, nElems)
            GGMLQuantizationType.Q8_1 -> dequantQ8_1(raw, nElems)
            GGMLQuantizationType.Q2_K -> dequantQ2K(raw, nElems)
            GGMLQuantizationType.Q3_K -> dequantQ3K(raw, nElems)
            GGMLQuantizationType.Q4_K -> dequantQ4K(raw, nElems)
            GGMLQuantizationType.Q5_K -> dequantQ5K(raw, nElems)
            GGMLQuantizationType.Q6_K -> dequantQ6K(raw, nElems)
            GGMLQuantizationType.Q8_K -> dequantQ8K(raw, nElems)
            GGMLQuantizationType.IQ4_NL -> dequantIQ4NL(raw, nElems)
            GGMLQuantizationType.IQ4_XS -> dequantIQ4XS(raw, nElems)
            GGMLQuantizationType.TQ1_0 -> dequantTQ1_0(raw, nElems)
            GGMLQuantizationType.TQ2_0 -> dequantTQ2_0(raw, nElems)
            else -> error("Dequantization for $tensorType not implemented")
        }
    }

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
     * Create an FP32 tensor from float data, transposing 2D tensors from column-major to row-major.
     * GGUF stores 2D tensors in column-major order, so we transpose them at load time.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> createFp32Tensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        originalShape: Shape,
        data: FloatArray
    ): Tensor<T, V> {
        return if (originalShape.rank == 2) {
            // Transpose 2D tensors from column-major to row-major
            val rows = originalShape[0]
            val cols = originalShape[1]
            val transposed = transposeColumnMajorToRowMajor(data, rows, cols)
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
                createFp32Tensor(ctx, dtype, shape, floats.toFloatArray())
            }

            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "F16/BF16 tensor ${rt.name} requires dtype Int8 with quantPolicy=RAW_BYTES; got ${dtype.simpleName}"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = when (val first = raw.firstOrNull()) {
                            is Byte -> raw.filterIsInstance<Byte>().toByteArray()
                            is UByte -> raw.filterIsInstance<UByte>().toUByteArray().toByteArray()
                            else -> error("Unexpected raw data type ${typeName(first)} for tensor ${rt.name}")
                        }
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }

                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        require(dtype == FP32::class) {
                            "Dequantizing ${rt.tensorType} to FP32 requires dtype FP32; got ${dtype.simpleName}"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val floats = when (rt.tensorType) {
                            GGMLQuantizationType.F16 -> dequantF16(raw)
                            GGMLQuantizationType.BF16 -> dequantBF16(raw)
                            else -> error("Unsupported native type ${rt.tensorType}")
                        }
                        createFp32Tensor(ctx, dtype, shape, floats)
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
                        val bytes: ByteArray = toByteArray(raw, rt.name)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }

                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        require(dtype == FP32::class) {
                            "Dequantizing ${rt.tensorType} to FP32 requires dtype FP32; got ${dtype.simpleName}"
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
                        createFp32Tensor(ctx, dtype, shape, floats)
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
                        val bytes: ByteArray = toByteArray(raw, rt.name)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        // Cannot dequantize unknown type - fall back to raw bytes with warning
                        println("WARNING: Cannot dequantize unknown type (raw: ${rt.rawTypeValue}) for '${rt.name}'. Falling back to raw bytes.")
                        require(dtype == Int8::class) {
                            "Unknown tensor type cannot be dequantized. Use dtype Int8 or add support for type ${rt.rawTypeValue}"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = toByteArray(raw, rt.name)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                }
            }

            else -> {
                // Fallback for any other unhandled types (shouldn't normally reach here)
                println("WARNING: Unhandled tensor type ${rt.tensorType} for '${rt.name}'. Storing as raw bytes.")
                require(dtype == Int8::class) {
                    "Unhandled tensor type ${rt.tensorType} requires dtype Int8; got ${dtype.simpleName}"
                }
                val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                val bytes: ByteArray = toByteArray(raw, rt.name)
                @Suppress("UNCHECKED_CAST")
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }
}
