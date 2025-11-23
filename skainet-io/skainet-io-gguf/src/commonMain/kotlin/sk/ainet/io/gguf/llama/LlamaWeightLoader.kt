package sk.ainet.io.gguf.llama

import kotlinx.io.Source
import kotlinx.io.buffered
import kotlinx.io.readByteArray
import kotlinx.io.readIntLe
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderField
import sk.ainet.io.gguf.QK_K
import sk.ainet.io.gguf.ReaderTensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.math.pow
import kotlin.math.max
import kotlin.ExperimentalUnsignedTypes
import kotlin.math.abs
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
 * Adapter that loads LLaMA weights either from GGUF (preferred) or Karpathy-style .bin
 * checkpoints and emits them in the canonical gguf tensor naming scheme. Validation covers
 * metadata presence and basic shape consistency for the tensors we materialize.
 */
public class LlamaWeightLoader(
    private val sourceProvider: () -> Source,
    private val format: Format = Format.GGUF,
    private val loadTensorData: Boolean = true,
    private val quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
    // Note: set loadTensorData=false to only validate metadata; tensors will be materialized
    // lazily when needed.
) {
    public enum class Format { GGUF, KARPATHY_BIN }
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
        @OptIn(ExperimentalUnsignedTypes::class)
        private fun toByteArray(raw: List<Any>, tensorName: String): ByteArray {
            val first = raw.firstOrNull()
            return when (first) {
                is Byte -> ByteArray(raw.size) { (raw[it] as Number).toByte() }
                is UByte -> UByteArray(raw.size) { (raw[it] as Number).toByte().toUByte() }.toByteArray()
                else -> error("Unexpected raw data type ${first?.javaClass} for tensor $tensorName")
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
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
                offset += 2
                for (j in 0 until 16) {
                    val b = bytes[offset + j].toInt() and 0xFF
                    val lo = b and 0x0F
                    val hi = b shr 4
                    out[outOff + j] = (lo - 8) * d
                    out[outOff + 16 + j] = (hi - 8) * d
                }
                offset += 16
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ5_0(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q5_0")
            val blockSize = 32
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
                val m = halfToFloat((bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF))
                offset += 4
                for (j in 0 until 32) {
                    out[outOff + j] = d * bytes[offset + j].toFloat() + m
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
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
                    val min = dMin * (minIdx / 15.0f)
                    repeat(16) { j ->
                        val codeByte = qs[block * 4 + j / 4].toInt() and 0xFF
                        val q = (codeByte ushr ((j % 4) * 2)) and 0x03
                        out[outOff + block * 16 + j] = q * scale + min
                    }
                }
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ3K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q3_K")
            val blockSize = QK_K
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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

        internal fun dequantQ4K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q4_K")
            val blockSize = QK_K
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
                repeat(8) { block ->
                    val bitPos = block * 12
                    val bytePos = bitPos / 8
                    val bitShift = bitPos % 8
                    val packed =
                        (scales.getOrElse(bytePos) { 0 }.toInt() and 0xFF) or
                            ((scales.getOrElse(bytePos + 1) { 0 }.toInt() and 0xFF) shl 8) or
                            ((scales.getOrElse(bytePos + 2) { 0 }.toInt() and 0xFF) shl 16)
                    val scaleIdx = (packed ushr bitShift) and 0x3F
                    val minIdx = (packed ushr (bitShift + 6)) and 0x3F
                    val scale = d * (scaleIdx / 63.0f)
                    val min = dMin * (minIdx / 63.0f)
                    repeat(32) { j ->
                        val codeByte = qs[block * 16 + j / 2].toInt() and 0xFF
                        val q = if (j % 2 == 0) codeByte and 0x0F else codeByte ushr 4
                        out[outOff + block * 32 + j] = q * scale + min
                    }
                }
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ5K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q5_K")
            val blockSize = QK_K
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
                repeat(8) { block ->
                    val bitPos = block * 12
                    val bytePos = bitPos / 8
                    val bitShift = bitPos % 8
                    val packed =
                        (scales.getOrElse(bytePos) { 0 }.toInt() and 0xFF) or
                            ((scales.getOrElse(bytePos + 1) { 0 }.toInt() and 0xFF) shl 8) or
                            ((scales.getOrElse(bytePos + 2) { 0 }.toInt() and 0xFF) shl 16)
                    val scaleIdx = (packed ushr bitShift) and 0x3F
                    val minIdx = (packed ushr (bitShift + 6)) and 0x3F
                    val scale = d * (scaleIdx / 63.0f)
                    val min = dMin * (minIdx / 63.0f)
                    repeat(32) { j ->
                        val idx = block * 32 + j
                        val low = qs[block * 16 + j / 2].toInt() and 0xFF
                        val qLow = if (j % 2 == 0) low and 0x0F else low ushr 4
                        val qHigh = (qh[idx / 8].toInt() ushr (idx % 8)) and 0x01
                        val q = qLow or (qHigh shl 4)
                        out[outOff + idx] = q * scale + min
                    }
                }
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ6K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q6_K")
            val blockSize = QK_K
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
            val out = FloatArray(blockCount * blockSize)
            var offset = 0
            var outOff = 0
            repeat(blockCount) {
                val d = halfToFloat(
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
                )
                offset += 2
                val scales = bytes.copyOfRange(offset, offset + 16)
                offset += 16
                val ql = bytes.copyOfRange(offset, offset + 128)
                offset += 128
                val qh = bytes.copyOfRange(offset, offset + 64)
                offset += 64
                repeat(16) { block ->
                    val scaleIdx = scales[block].toInt() and 0xFF
                    val scale = d * (scaleIdx / 127.0f)
                    repeat(16) { j ->
                        val idx = block * 16 + j
                        val lowByte = ql[idx / 2].toInt() and 0xFF
                        val qLow = if (idx % 2 == 0) lowByte and 0x0F else lowByte ushr 4
                        val qHigh = (qh[idx / 4].toInt() ushr ((idx % 4) * 2)) and 0x03
                        val q = qLow or (qHigh shl 4)
                        out[outOff + idx] = q * scale
                    }
                }
                outOff += blockSize
            }
            return out
        }

        internal fun dequantQ8K(raw: List<Any>, nElems: Int): FloatArray {
            val bytes = toByteArray(raw, "Q8_K")
            val blockSize = QK_K
            val blockCount = max(1, (nElems + blockSize - 1) / blockSize)
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
    }

    /**
     * Load weights and invoke [onTensorLoaded] for each required tensor. Returns parsed metadata.
     */
    public suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): LlamaModelMetadata {
        return when (format) {
            Format.GGUF -> loadFromGguf(ctx, dtype, onTensorLoaded, null)
            Format.KARPATHY_BIN -> loadFromKarpathyBin(ctx, dtype, onTensorLoaded, null)
        }
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
        val meta = when (format) {
            Format.GGUF -> loadFromGguf(ctx, dtype, { name, tensor -> byName[name] = tensor }) { name, qt ->
                quantTypes[name] = qt
            }

            Format.KARPATHY_BIN -> loadFromKarpathyBin(ctx, dtype, { name, tensor -> byName[name] = tensor }) { name, qt ->
                quantTypes[name] = qt
            }
        }
        return LlamaWeights(meta, byName, quantTypes)
    }

    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): LlamaWeights<T, V> = loadToMap(ctx, T::class)

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?
    ): LlamaModelMetadata {
        require(dtype == FP32::class) {
            "LLaMA GGUF loader currently supports FP32 tensors only (got ${dtype.simpleName})"
        }

        val reader = sourceProvider().buffered().use { src ->
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

    private fun <T : DType, V> loadFromKarpathyBin(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?
    ): LlamaModelMetadata {
        require(dtype == FP32::class) {
            "Karpathy .bin loader currently supports FP32 tensors only (got ${dtype.simpleName})"
        }

        return sourceProvider().buffered().use { buffer ->
            val metadata = readKarpathyMetadata(buffer)
            validateMetadata(metadata)

            val headSize = metadata.embeddingLength / metadata.headCount

            // Token embeddings
            val tokenEmbeddings = buffer.readFloatLeArray(metadata.vocabSize * metadata.embeddingLength)
            onTensorLoaded(
                LlamaTensorNames.TOKEN_EMBEDDINGS,
                ctx.fromFloatArray<T, Float>(
                    Shape(metadata.vocabSize, metadata.embeddingLength),
                    dtype,
                    tokenEmbeddings
                ) as Tensor<T, V>
            )

            // Layered weights
            val baseDim = metadata.embeddingLength
            val ffDim = metadata.feedForwardLength
            val layerMatSize = baseDim * baseDim
            val gateMatSize = ffDim * baseDim
            val downMatSize = baseDim * ffDim

            val rmsAtt = buffer.readFloatLeArray(metadata.blockCount * baseDim)
            val wq = buffer.readFloatLeArray(metadata.blockCount * layerMatSize)
            val wk = buffer.readFloatLeArray(metadata.blockCount * layerMatSize)
            val wv = buffer.readFloatLeArray(metadata.blockCount * layerMatSize)
            val wo = buffer.readFloatLeArray(metadata.blockCount * layerMatSize)
            val rmsFfn = buffer.readFloatLeArray(metadata.blockCount * baseDim)
            val w1 = buffer.readFloatLeArray(metadata.blockCount * gateMatSize)
            val w2 = buffer.readFloatLeArray(metadata.blockCount * downMatSize)
            val w3 = buffer.readFloatLeArray(metadata.blockCount * gateMatSize)

            repeat(metadata.blockCount) { layer ->
                onTensorLoaded(
                    LlamaTensorNames.attnNorm(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(baseDim),
                    dtype,
                    sliceFloats(rmsAtt, layer * baseDim, baseDim)
                ) as Tensor<T, V>
            )

                onTensorLoaded(
                    LlamaTensorNames.attnQ(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(baseDim, baseDim),
                    dtype,
                    sliceFloats(wq, layer * baseDim * baseDim, baseDim * baseDim)
                ) as Tensor<T, V>
            )

                onTensorLoaded(
                    LlamaTensorNames.attnK(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(baseDim, baseDim),
                    dtype,
                    sliceFloats(wk, layer * baseDim * baseDim, baseDim * baseDim)
                ) as Tensor<T, V>
            )

                onTensorLoaded(
                    LlamaTensorNames.attnV(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(baseDim, baseDim),
                    dtype,
                    sliceFloats(wv, layer * baseDim * baseDim, baseDim * baseDim)
                ) as Tensor<T, V>
            )

                onTensorLoaded(
                    LlamaTensorNames.attnOut(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(baseDim, baseDim),
                    dtype,
                    sliceFloats(wo, layer * baseDim * baseDim, baseDim * baseDim)
                ) as Tensor<T, V>
            )

                onTensorLoaded(
                    LlamaTensorNames.ffnNorm(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(baseDim),
                    dtype,
                    sliceFloats(rmsFfn, layer * baseDim, baseDim)
                ) as Tensor<T, V>
            )

                onTensorLoaded(
                    LlamaTensorNames.ffnGate(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(ffDim, baseDim),
                    dtype,
                    sliceFloats(w1, layer * ffDim * baseDim, ffDim * baseDim)
                ) as Tensor<T, V>
            )

                onTensorLoaded(
                    LlamaTensorNames.ffnDown(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(baseDim, ffDim),
                    dtype,
                    sliceFloats(w2, layer * baseDim * ffDim, baseDim * ffDim)
                ) as Tensor<T, V>
            )

                onTensorLoaded(
                    LlamaTensorNames.ffnUp(layer),
                ctx.fromFloatArray<T, Float>(
                    Shape(ffDim, baseDim),
                    dtype,
                    sliceFloats(w3, layer * ffDim * baseDim, ffDim * baseDim)
                ) as Tensor<T, V>
            )
            }

            // Final norm
            val rmsFinal = buffer.readFloatLeArray(metadata.embeddingLength)
            onTensorLoaded(
                LlamaTensorNames.OUTPUT_NORM,
                ctx.fromFloatArray<T, Float>(
                    Shape(metadata.embeddingLength),
                    dtype,
                    rmsFinal
                ) as Tensor<T, V>
            )

            // RoPE tables (kept optional and not emitted as tensors yet)
            val _freqCisReal = buffer.readFloatLeArray(metadata.contextLength * headSize / 2)
            val _freqCisImag = buffer.readFloatLeArray(metadata.contextLength * headSize / 2)
            onTensorLoaded(
                LlamaTensorNames.ROPE_FREQS_REAL,
                ctx.fromFloatArray<T, Float>(
                    Shape(metadata.contextLength, headSize / 2),
                    dtype,
                    _freqCisReal
                ) as Tensor<T, V>
            )
            onTensorLoaded(
                LlamaTensorNames.ROPE_FREQS_IMAG,
                ctx.fromFloatArray<T, Float>(
                    Shape(metadata.contextLength, headSize / 2),
                    dtype,
                    _freqCisImag
                ) as Tensor<T, V>
            )

            // Classifier / output weight
            onTensorLoaded(
                LlamaTensorNames.OUTPUT_WEIGHT,
                ctx.fromFloatArray<T, Float>(
                    Shape(metadata.vocabSize, metadata.embeddingLength),
                    dtype,
                    tokenEmbeddings
                ) as Tensor<T, V>
            )

            metadata
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

                    name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v")
                            || name.contains("attn_output") -> {
                        require(dims.size == 2 && dims.all { it == metadata.embeddingLength }) {
                            "Tensor $name must be [dim, dim]; got $dims"
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
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        val dims = token.shape.map { it.toInt() }
        val emb = dims.maxOrNull() ?: 0
        val prod = dims.product()
        val vocab = if (emb == 0) 0 else prod / emb
        require(vocab > 0) { "Cannot infer vocab size from shape $dims" }
        return vocab
    }

    private fun List<Int>.product(): Int = fold(1) { acc, v -> acc * v }

    private fun bytesToFloat(bytes: ByteArray, littleEndian: Boolean = true): Float {
        val bits = if (littleEndian) {
            bytes[0].toInt() and 0xFF or
                (bytes[1].toInt() and 0xFF shl 8) or
                (bytes[2].toInt() and 0xFF shl 16) or
                (bytes[3].toInt() shl 24)
        } else {
            bytes[3].toInt() and 0xFF or
                (bytes[2].toInt() and 0xFF shl 8) or
                (bytes[1].toInt() and 0xFF shl 16) or
                (bytes[0].toInt() shl 24)
        }
        return Float.fromBits(bits)
    }

    private fun Source.readFloatLe(): Float = bytesToFloat(readByteArray(4))

    private fun Source.readFloatLeArray(size: Int): FloatArray {
        val floats = FloatArray(size)
        for (i in 0 until size) {
            floats[i] = readFloatLe()
        }
        return floats
    }

    private fun sliceFloats(src: FloatArray, offset: Int, length: Int): FloatArray {
        val out = FloatArray(length)
        src.copyInto(out, 0, offset, offset + length)
        return out
    }

    private fun readKarpathyMetadata(buffer: Source): LlamaModelMetadata {
        val dim = buffer.readIntLe()
        val hiddenDim = buffer.readIntLe()
        val nLayers = buffer.readIntLe()
        val nHeads = buffer.readIntLe()
        val nKvHeads = buffer.readIntLe()
        val vocabSize = abs(buffer.readIntLe())
        val seqLen = buffer.readIntLe()

        return LlamaModelMetadata(
            architecture = "llama",
            embeddingLength = dim,
            contextLength = seqLen,
            blockCount = nLayers,
            headCount = nHeads,
            kvHeadCount = nKvHeads,
            feedForwardLength = hiddenDim,
            ropeDimensionCount = dim / nHeads,
            vocabSize = vocabSize
        )
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
                ctx.fromFloatArray<T, Float>(shape, dtype, floats.toFloatArray()) as Tensor<T, V>
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
                            else -> error("Unexpected raw data type ${first?.javaClass} for tensor ${rt.name}")
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
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
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
            GGMLQuantizationType.IQ4_XS -> {
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
                            else -> error("Dequantization for ${rt.tensorType} not implemented yet")
                        }
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
                    }
                }
            }

            else -> {
                // Fallback: keep raw bytes even if DEQUANT policy was requested.
                if (quantPolicy == QuantPolicy.DEQUANTIZE_TO_FP32) {
                    // Optionally log/trace; for now, fall back to RAW_BYTES.
                }
                require(dtype == Int8::class) {
                    "Quantized tensor ${rt.name} requires dtype Int8 with quantPolicy=RAW_BYTES; got ${dtype.simpleName}"
                }
                val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                val bytes: ByteArray = toByteArray(raw, rt.name)
                @Suppress("UNCHECKED_CAST")
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }
}
