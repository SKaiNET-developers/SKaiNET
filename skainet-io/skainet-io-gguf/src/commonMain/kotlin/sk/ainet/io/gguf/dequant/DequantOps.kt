package sk.ainet.io.gguf.dequant

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.QK_K
import kotlin.math.pow

/**
 * Shared dequantization operations for GGUF and SafeTensors weight loading.
 *
 * All functions are module-visible (`internal`) and stateless. They convert
 * quantized byte payloads into FP32 float arrays.
 */
public object DequantOps {

    // ========== List<Any>-based variants (GGUFReader in-memory) ==========

    @OptIn(ExperimentalUnsignedTypes::class)
    public fun toByteArray(raw: List<Any>, tensorName: String): ByteArray {
        val first = raw.firstOrNull()
        return when (first) {
            is Byte -> ByteArray(raw.size) { (raw[it] as Number).toByte() }
            is UByte -> ByteArray(raw.size) { (raw[it] as UByte).toByte() }
            else -> error("Unexpected raw data type ${typeName(first)} for tensor $tensorName")
        }
    }

    public fun dequantF16(raw: List<Any>): FloatArray {
        val bytes: ByteArray = toByteArray(raw, "F16")
        return dequantF16FromBytes(bytes)
    }

    public fun dequantBF16(raw: List<Any>): FloatArray {
        val bytes: ByteArray = toByteArray(raw, "BF16")
        return dequantBF16FromBytes(bytes)
    }

    public fun dequantQ4_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q4_0")
        return dequantQ4_0FromBytes(bytes, nElems)
    }

    public fun dequantQ5_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q5_0")
        return dequantQ5_0FromBytes(bytes, nElems)
    }

    public fun dequantQ8_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q8_0")
        return dequantQ8_0FromBytes(bytes, nElems)
    }

    public fun dequantQ4_1(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q4_1")
        return dequantQ4_1FromBytes(bytes, nElems)
    }

    public fun dequantQ5_1(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q5_1")
        return dequantQ5_1FromBytes(bytes, nElems)
    }

    public fun dequantQ8_1(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q8_1")
        return dequantQ8_1FromBytes(bytes, nElems)
    }

    public fun dequantIQ4NL(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ4_NL")
        return dequantIQ4NLFromBytes(bytes, nElems)
    }

    public fun dequantIQ4XS(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ4_XS")
        return dequantIQ4XSFromBytes(bytes, nElems)
    }

    public fun dequantMXFP4(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "MXFP4")
        return dequantMXFP4FromBytes(bytes, nElems)
    }

    public fun dequantIQ2XXS(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ2_XXS")
        return dequantIQ2XXSFromBytes(bytes, nElems)
    }

    public fun dequantIQ2XS(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ2_XS")
        return dequantIQ2XSFromBytes(bytes, nElems)
    }

    public fun dequantIQ2S(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ2_S")
        return dequantIQ2SFromBytes(bytes, nElems)
    }

    public fun dequantIQ3XXS(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ3_XXS")
        return dequantIQ3XXSFromBytes(bytes, nElems)
    }

    public fun dequantIQ3S(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ3_S")
        return dequantIQ3SFromBytes(bytes, nElems)
    }

    public fun dequantNVFP4(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "NVFP4")
        return dequantNVFP4FromBytes(bytes, nElems)
    }

    public fun dequantQ2K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q2_K")
        return dequantQ2KFromBytes(bytes, nElems)
    }

    public fun dequantQ3K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q3_K")
        return dequantQ3KFromBytes(bytes, nElems)
    }

    public fun dequantQ4K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q4_K")
        return dequantQ4KFromBytes(bytes, nElems)
    }

    public fun dequantQ5K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q5_K")
        return dequantQ5KFromBytes(bytes, nElems)
    }

    public fun dequantQ6K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q6_K")
        return dequantQ6KFromBytes(bytes, nElems)
    }

    public fun dequantQ8K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q8_K")
        return dequantQ8KFromBytes(bytes, nElems)
    }

    public fun dequantTQ2_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "TQ2_0")
        return dequantTQ2_0FromBytes(bytes, nElems)
    }

    public fun dequantTQ1_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "TQ1_0")
        return dequantTQ1_0FromBytes(bytes, nElems)
    }

    // ========== ByteArray-based variants (StreamingGGUFReader / SafeTensors) ==========

    public fun dequantF16FromBytes(bytes: ByteArray): FloatArray {
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

    public fun dequantBF16FromBytes(bytes: ByteArray): FloatArray {
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

    public fun bytesToFloatArray(bytes: ByteArray): FloatArray {
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

    public fun halfToFloat(hbits: Int): Float {
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
     * Handle column-major to row-major conversion for GGUF tensors.
     * Data layout is unchanged — only the shape interpretation changes.
     */
    @Suppress("UNUSED_PARAMETER")
    public fun transposeColumnMajorToRowMajor(
        data: FloatArray,
        rows: Int,
        cols: Int
    ): FloatArray = data

    /**
     * Dispatch dequantization based on tensor type for byte arrays.
     */
    fun dequantFromBytes(bytes: ByteArray, tensorType: GGMLQuantizationType, nElems: Int): FloatArray {
        return when (tensorType) {
            GGMLQuantizationType.F16 -> dequantF16FromBytes(bytes)
            GGMLQuantizationType.BF16 -> dequantBF16FromBytes(bytes)
            GGMLQuantizationType.Q4_0 -> dequantQ4_0FromBytes(bytes, nElems)
            GGMLQuantizationType.Q4_1 -> dequantQ4_1FromBytes(bytes, nElems)
            GGMLQuantizationType.Q5_0 -> dequantQ5_0FromBytes(bytes, nElems)
            GGMLQuantizationType.Q5_1 -> dequantQ5_1FromBytes(bytes, nElems)
            GGMLQuantizationType.Q8_0 -> dequantQ8_0FromBytes(bytes, nElems)
            GGMLQuantizationType.Q8_1 -> dequantQ8_1FromBytes(bytes, nElems)
            GGMLQuantizationType.Q2_K -> dequantQ2KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q3_K -> dequantQ3KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q4_K -> dequantQ4KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q5_K -> dequantQ5KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q6_K -> dequantQ6KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q8_K -> dequantQ8KFromBytes(bytes, nElems)
            GGMLQuantizationType.IQ4_NL -> dequantIQ4NLFromBytes(bytes, nElems)
            GGMLQuantizationType.IQ4_XS -> dequantIQ4XSFromBytes(bytes, nElems)
            GGMLQuantizationType.TQ1_0 -> dequantTQ1_0FromBytes(bytes, nElems)
            GGMLQuantizationType.TQ2_0 -> dequantTQ2_0FromBytes(bytes, nElems)
            GGMLQuantizationType.MXFP4 -> dequantMXFP4FromBytes(bytes, nElems)
            GGMLQuantizationType.IQ2_XXS -> dequantIQ2XXSFromBytes(bytes, nElems)
            GGMLQuantizationType.IQ2_XS -> dequantIQ2XSFromBytes(bytes, nElems)
            GGMLQuantizationType.IQ2_S -> dequantIQ2SFromBytes(bytes, nElems)
            GGMLQuantizationType.IQ3_XXS -> dequantIQ3XXSFromBytes(bytes, nElems)
            GGMLQuantizationType.IQ3_S -> dequantIQ3SFromBytes(bytes, nElems)
            GGMLQuantizationType.NVFP4 -> dequantNVFP4FromBytes(bytes, nElems)
            else -> error("Dequantization for $tensorType not implemented")
        }
    }

    /**
     * Dispatch dequantization based on tensor type for List<Any> (in-memory GGUFReader).
     */
    public fun dequantFromList(raw: List<Any>, tensorType: GGMLQuantizationType, nElems: Int): FloatArray {
        return when (tensorType) {
            GGMLQuantizationType.F16 -> dequantF16(raw)
            GGMLQuantizationType.BF16 -> dequantBF16(raw)
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
            GGMLQuantizationType.MXFP4 -> dequantMXFP4(raw, nElems)
            GGMLQuantizationType.IQ2_XXS -> dequantIQ2XXS(raw, nElems)
            GGMLQuantizationType.IQ2_XS -> dequantIQ2XS(raw, nElems)
            GGMLQuantizationType.IQ2_S -> dequantIQ2S(raw, nElems)
            GGMLQuantizationType.IQ3_XXS -> dequantIQ3XXS(raw, nElems)
            GGMLQuantizationType.IQ3_S -> dequantIQ3S(raw, nElems)
            GGMLQuantizationType.NVFP4 -> dequantNVFP4(raw, nElems)
            else -> error("Dequantization for $tensorType not implemented")
        }
    }

    /**
     * Returns (bytesPerBlock, elemsPerBlock) for a given quantization type.
     * Useful for chunked dequantization on single-threaded platforms (WASM).
     */
    public fun blockInfoFor(type: GGMLQuantizationType): Pair<Int, Int> = when (type) {
        GGMLQuantizationType.F16  -> Pair(2, 1)
        GGMLQuantizationType.BF16 -> Pair(2, 1)
        GGMLQuantizationType.F32  -> Pair(4, 1)
        GGMLQuantizationType.Q4_0 -> Pair(18, 32)
        GGMLQuantizationType.Q4_1 -> Pair(20, 32)
        GGMLQuantizationType.Q5_0 -> Pair(22, 32)
        GGMLQuantizationType.Q5_1 -> Pair(24, 32)
        GGMLQuantizationType.Q8_0 -> Pair(34, 32)
        GGMLQuantizationType.Q8_1 -> Pair(36, 32)
        GGMLQuantizationType.IQ4_NL -> Pair(18, 32)
        GGMLQuantizationType.IQ4_XS -> Pair(2 + 2 + QK_K / 2 + QK_K / 64, QK_K)
        GGMLQuantizationType.Q2_K -> Pair(2 + 2 + QK_K / 16 + QK_K / 4, QK_K)
        GGMLQuantizationType.Q3_K -> Pair(2 + QK_K / 4 + QK_K / 8 + 12, QK_K)
        GGMLQuantizationType.Q4_K -> Pair(144, QK_K)
        GGMLQuantizationType.Q5_K -> Pair(176, QK_K)
        GGMLQuantizationType.Q6_K -> Pair(210, QK_K)
        GGMLQuantizationType.Q8_K -> Pair(292, QK_K)
        GGMLQuantizationType.TQ1_0 -> Pair(54, 256)
        GGMLQuantizationType.TQ2_0 -> Pair(66, 256)
        GGMLQuantizationType.MXFP4 -> Pair(17, 32)
        GGMLQuantizationType.IQ2_XXS -> Pair(2 + QK_K / 4, QK_K)
        GGMLQuantizationType.IQ2_XS -> Pair(2 + QK_K / 4 + QK_K / 32, QK_K)
        GGMLQuantizationType.IQ2_S -> Pair(2 + QK_K / 4 + QK_K / 16, QK_K)
        GGMLQuantizationType.IQ3_XXS -> Pair(2 + 3 * QK_K / 8, QK_K)
        GGMLQuantizationType.IQ3_S -> Pair(2 + 13 * QK_K / 32 + QK_K / 64, QK_K)
        GGMLQuantizationType.NVFP4 -> Pair(4 + 32, 64)
        else -> error("Block info for $type not available")
    }

    // ========== ByteArray-based quantization implementations ==========

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ4_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 18
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
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

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ5_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 22
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

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ8_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 34
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

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ4_1FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 20
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

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ5_1FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 24
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

    /**
     * `Q8_1`: 36 bytes for 32 elements — fp16 `d`, fp16 `s` (= `d` times the sum of the quants),
     * then 32 signed quant bytes; `y = d * q`. ggml never writes Q8_1 into GGUF files (it is a
     * dot-product intermediate) and has no `dequantize_row_q8_1`, so unlike the other types this
     * decoder has no llama.cpp fixture to be held to; the layout is `block_q8_1` of
     * `ggml-common.h`. (The previous decoder assumed a pre-2024 40-byte block with fp32 fields.)
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ8_1FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 2 + 2 + 32
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            offset += 4 // skip d and s, fp16 each
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

    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ4NLFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 18
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

    /**
     * `MXFP4` (#1410): 17 bytes for 32 elements. Byte 0 is the shared E8M0 exponent `e`, the
     * other 16 bytes hold two 4-bit codes each; the low nibbles fill the first half of the block
     * and the high nibbles the second half. `kvalues_mxfp4` holds twice the E2M1 value, so the
     * scale is half of `2^(e - 127)`: the float with bits `(e - 1) shl 23`, and the two
     * subnormals `2^-128` / `2^-127` for `e` = 0 / 1. Bit-identical to llama.cpp's
     * `dequantize_row_mxfp4` (see the reference page *GGUF block layouts: IQ and MXFP4*).
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantMXFP4FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 17
        val values = GgufIqTables.KVALUES_MXFP4
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val e = bytes[offset].toInt() and 0xFF
            val d = Float.fromBits(if (e < 2) 0x00200000 shl e else (e - 1) shl 23)
            offset += 1
            for (j in 0 until blockSize / 2) {
                val code = bytes[offset + j].toInt() and 0xFF
                out[outOff + j] = values[code and 0x0F] * d
                out[outOff + blockSize / 2 + j] = values[code ushr 4] * d
            }
            offset += blockSize / 2
            outOff += blockSize
        }
        return out
    }

    /**
     * `UE4M3` → FP32: ggml's unsigned 8-bit scale format for NVFP4 (4 exponent bits, bias 7,
     * 3 mantissa bits; no sign — bit 7 is ignored). `0x00` and `0x7F` (the E4M3 NaN pattern)
     * decode to 0. The result is halved because `kvalues_mxfp4` holds twice the E2M1 value.
     * Mirrors `ggml_ue4m3_to_fp32`; every step is exact in FP32.
     */
    private fun ue4m3ToFloat(x: Int): Float {
        if (x == 0 || x == 0x7F) return 0.0f
        val exp = (x ushr 3) and 0xF
        val man = x and 0x7
        val raw = if (exp == 0) {
            man.toFloat() * Float.fromBits((127 - 9) shl 23) // man * 2^-9
        } else {
            (1.0f + man.toFloat() / 8.0f) * Float.fromBits((127 + exp - 7) shl 23) // * 2^(exp-7)
        }
        return raw * 0.5f
    }

    /**
     * `NVFP4`: 36 bytes for 64 elements — four 16-element sub-blocks, each with one unsigned
     * E4M3 scale byte (`d[4]`), then 32 bytes of E2M1 nibbles over [GgufIqTables.KVALUES_MXFP4]
     * (low nibbles fill a sub-block's first half, high nibbles the second). Bit-identical to
     * llama.cpp's `dequantize_row_nvfp4`.
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantNVFP4FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val bytesPerBlock = 4 + 32
        val values = GgufIqTables.KVALUES_MXFP4
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * 64)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            for (s in 0 until 4) {
                val d = ue4m3ToFloat(bytes[offset + s].toInt() and 0xFF)
                val qs = offset + 4 + s * 8
                val yb = outOff + s * 16
                for (j in 0 until 8) {
                    val code = bytes[qs + j].toInt() and 0xFF
                    out[yb + j] = values[code and 0x0F] * d
                    out[yb + 8 + j] = values[code ushr 4] * d
                }
            }
            offset += bytesPerBlock
            outOff += 64
        }
        return out
    }

    /**
     * Writes one IQ sub-group of 8 values: `out[outOff + j] = db * grid[gridOff + j]`, negated
     * where bit `j` of [signs] is set (`kmask_iq2xs[j]` is `1 shl j`). The grid bytes are
     * unsigned. The product is formed first and the sign applied afterwards, as llama.cpp does.
     */
    private fun writeIqSubGroup(out: FloatArray, outOff: Int, db: Float, grid: ByteArray, gridOff: Int, signs: Int) {
        for (j in 0 until 8) {
            val v = db * (grid[gridOff + j].toInt() and 0xFF)
            out[outOff + j] = if ((signs ushr j) and 1 != 0) -v else v
        }
    }

    /**
     * `IQ2_XXS` (#1411): 66 bytes for 256 elements. An fp16 scale `d`, then 8 bytes per group
     * of 32, read as two little-endian `uint32`: the four bytes of the first are the
     * `iq2xxs_grid` indices of the four sub-groups; the second holds four 7-bit indices into
     * `ksigns_iq2xs` and, in its top 4 bits, the group scale. Bit-identical to llama.cpp's
     * `dequantize_row_iq2_xxs` (see the reference page *GGUF block layouts: IQ and MXFP4*).
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ2XXSFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val bytesPerBlock = 2 + QK_K / 4
        val grid = GgufIqTables.IQ2XXS_GRID
        val ksigns = GgufIqTables.KSIGNS_IQ2XS
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * QK_K)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            val qs = offset + 2
            for (ib in 0 until QK_K / 32) {
                val g = qs + 8 * ib
                val a1 = (bytes[g + 4].toInt() and 0xFF) or
                    (bytes[g + 5].toInt() and 0xFF shl 8) or
                    (bytes[g + 6].toInt() and 0xFF shl 16) or
                    (bytes[g + 7].toInt() and 0xFF shl 24)
                val db = d * (0.5f + (a1 ushr 28)) * 0.25f
                for (l in 0 until 4) {
                    val index = bytes[g + l].toInt() and 0xFF
                    val signs = ksigns[(a1 ushr (7 * l)) and 127].toInt() and 0xFF
                    writeIqSubGroup(out, outOff, db, grid, 8 * index, signs)
                    outOff += 8
                }
            }
            offset += bytesPerBlock
        }
        return out
    }

    /**
     * `IQ2_XS` (#1411): 74 bytes for 256 elements. An fp16 scale `d`, 32 little-endian
     * `uint16` (one per sub-group: 9 bits of `iq2xs_grid` index, 7 bits of `ksigns_iq2xs`
     * index), then one scale byte per group of 32 whose low nibble serves sub-groups 0 and 1
     * and whose high nibble serves 2 and 3. Bit-identical to llama.cpp's `dequantize_row_iq2_xs`.
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ2XSFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val bytesPerBlock = 2 + QK_K / 4 + QK_K / 32
        val grid = GgufIqTables.IQ2XS_GRID
        val ksigns = GgufIqTables.KSIGNS_IQ2XS
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * QK_K)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            val qs = offset + 2
            val scales = qs + QK_K / 4
            for (ib in 0 until QK_K / 32) {
                val scale = bytes[scales + ib].toInt() and 0xFF
                val dbLo = d * (0.5f + (scale and 0x0F)) * 0.25f
                val dbHi = d * (0.5f + (scale ushr 4)) * 0.25f
                for (l in 0 until 4) {
                    val p = qs + 2 * (4 * ib + l)
                    val q = (bytes[p].toInt() and 0xFF) or (bytes[p + 1].toInt() and 0xFF shl 8)
                    val signs = ksigns[q ushr 9].toInt() and 0xFF
                    writeIqSubGroup(out, outOff, if (l < 2) dbLo else dbHi, grid, 8 * (q and 511), signs)
                    outOff += 8
                }
            }
            offset += bytesPerBlock
        }
        return out
    }

    /**
     * `IQ2_S` (#1411): 82 bytes for 256 elements. An fp16 scale `d`, 32 bytes with the low 8
     * bits of the `iq2s_grid` index of each sub-group, 32 sign bytes (used directly, without
     * `ksigns_iq2xs`), 8 bytes `qh` with the 2 high index bits of each sub-group, and one scale
     * byte per group of 32 as in `IQ2_XS`. Bit-identical to llama.cpp's `dequantize_row_iq2_s`.
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ2SFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val bytesPerBlock = 2 + QK_K / 4 + QK_K / 16
        val grid = GgufIqTables.IQ2S_GRID
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * QK_K)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            val qs = offset + 2
            val signBytes = qs + QK_K / 8
            val qh = qs + QK_K / 4
            val scales = qh + QK_K / 32
            for (ib in 0 until QK_K / 32) {
                val scale = bytes[scales + ib].toInt() and 0xFF
                val dbLo = d * (0.5f + (scale and 0x0F)) * 0.25f
                val dbHi = d * (0.5f + (scale ushr 4)) * 0.25f
                val high = bytes[qh + ib].toInt() and 0xFF
                for (l in 0 until 4) {
                    val index = (bytes[qs + 4 * ib + l].toInt() and 0xFF) or (((high ushr (2 * l)) and 3) shl 8)
                    val signs = bytes[signBytes + 4 * ib + l].toInt() and 0xFF
                    writeIqSubGroup(out, outOff, if (l < 2) dbLo else dbHi, grid, 8 * index, signs)
                    outOff += 8
                }
            }
            offset += bytesPerBlock
        }
        return out
    }

    /**
     * Writes one IQ3 sub-group of 8 values from two grid entries of 4 unsigned bytes each:
     * elements 0..3 from [gridOff1], 4..7 from [gridOff2]; bit `j` of [signs] negates element `j`.
     */
    private fun writeIq3SubGroup(
        out: FloatArray, outOff: Int, db: Float, grid: ByteArray, gridOff1: Int, gridOff2: Int, signs: Int,
    ) {
        for (j in 0 until 4) {
            val v1 = db * (grid[gridOff1 + j].toInt() and 0xFF)
            val v2 = db * (grid[gridOff2 + j].toInt() and 0xFF)
            out[outOff + j] = if ((signs ushr j) and 1 != 0) -v1 else v1
            out[outOff + 4 + j] = if ((signs ushr (j + 4)) and 1 != 0) -v2 else v2
        }
    }

    /**
     * `IQ3_XXS` (#1412): 98 bytes for 256 elements. An fp16 scale `d`, 64 bytes of
     * `iq3xxs_grid` indices (two per sub-group, each entry gives 4 values), then one
     * little-endian `uint32` per group of 32 with four 7-bit `ksigns_iq2xs` indices and, in its
     * top 4 bits, the group scale. The final factor is `0.5f`, not the `0.25f` of the IQ2
     * formats. Bit-identical to llama.cpp's `dequantize_row_iq3_xxs`.
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ3XXSFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val bytesPerBlock = 2 + 3 * QK_K / 8
        val grid = GgufIqTables.IQ3XXS_GRID
        val ksigns = GgufIqTables.KSIGNS_IQ2XS
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * QK_K)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            val qs = offset + 2
            val scalesAndSigns = qs + QK_K / 4
            for (ib in 0 until QK_K / 32) {
                val p = scalesAndSigns + 4 * ib
                val a = (bytes[p].toInt() and 0xFF) or
                    (bytes[p + 1].toInt() and 0xFF shl 8) or
                    (bytes[p + 2].toInt() and 0xFF shl 16) or
                    (bytes[p + 3].toInt() and 0xFF shl 24)
                val db = d * (0.5f + (a ushr 28)) * 0.5f
                for (l in 0 until 4) {
                    val signs = ksigns[(a ushr (7 * l)) and 127].toInt() and 0xFF
                    val i1 = bytes[qs + 8 * ib + 2 * l].toInt() and 0xFF
                    val i2 = bytes[qs + 8 * ib + 2 * l + 1].toInt() and 0xFF
                    writeIq3SubGroup(out, outOff, db, grid, 4 * i1, 4 * i2, signs)
                    outOff += 8
                }
            }
            offset += bytesPerBlock
        }
        return out
    }

    /**
     * `IQ3_S` (#1412): 110 bytes for 256 elements. An fp16 scale `d`, 64 bytes with the low 8
     * bits of the `iq3s_grid` indices (two per sub-group), 8 bytes `qh` (one per group of 32,
     * the 9th bit of its 8 indices), 32 sign bytes (used directly) and 4 scale bytes, one per
     * pair of groups: low nibble for the even group, high nibble for the odd one. The group
     * scale is `d * (1 + 2*s)`. Bit-identical to llama.cpp's `dequantize_row_iq3_s`.
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ3SFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val bytesPerBlock = 2 + 13 * QK_K / 32 + QK_K / 64
        val grid = GgufIqTables.IQ3S_GRID
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * QK_K)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            val qs = offset + 2
            val qh = qs + QK_K / 4
            val signBytes = qh + QK_K / 32
            val scales = signBytes + QK_K / 8
            for (ib in 0 until QK_K / 32) {
                val scale = bytes[scales + ib / 2].toInt() and 0xFF
                val s = if (ib % 2 == 0) scale and 0x0F else scale ushr 4
                val db = d * (1 + 2 * s)
                val high = bytes[qh + ib].toInt() and 0xFF
                for (l in 0 until 4) {
                    val i1 = (bytes[qs + 8 * ib + 2 * l].toInt() and 0xFF) or (((high ushr (2 * l)) and 1) shl 8)
                    val i2 = (bytes[qs + 8 * ib + 2 * l + 1].toInt() and 0xFF) or (((high ushr (2 * l + 1)) and 1) shl 8)
                    val signs = bytes[signBytes + 4 * ib + l].toInt() and 0xFF
                    writeIq3SubGroup(out, outOff, db, grid, 4 * i1, 4 * i2, signs)
                    outOff += 8
                }
            }
            offset += bytesPerBlock
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ4XSFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 2 + 2 + QK_K / 2 + QK_K / 64
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

    /**
     * `Q2_K`: 84 bytes for 256 elements — `scales[16]` (low nibble: scale, high nibble: min),
     * `qs[64]`, then fp16 `d` and `dmin` at the end. Each 128-element half sweeps a 2-bit
     * `shift` over the same 32 `qs` bytes, two 16-element runs per shift. Bit-identical to
     * llama.cpp's `dequantize_row_q2_K`. (The previous decoder read `d`/`dmin` first, used the
     * high nibble as the scale and divided by 15 — every value came out wrong; caught by
     * `GgufQuantDequantReferenceTest`.)
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ2KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val bytesPerBlock = QK_K / 16 + QK_K / 4 + 2 + 2
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * QK_K)
        var o = 0
        for (i in 0 until blockCount) {
            val base = i * bytesPerBlock
            val scales = base
            val d = halfToFloat((bytes[base + 81].toInt() and 0xFF shl 8) or (bytes[base + 80].toInt() and 0xFF))
            val min = halfToFloat((bytes[base + 83].toInt() and 0xFF shl 8) or (bytes[base + 82].toInt() and 0xFF))
            var isIdx = 0
            var q = base + QK_K / 16
            for (half in 0 until 2) {
                var shift = 0
                for (j in 0 until 4) {
                    var sc = bytes[scales + isIdx++].toInt() and 0xFF
                    var dl = d * (sc and 0xF)
                    var ml = min * (sc ushr 4)
                    for (l in 0 until 16) {
                        out[o++] = dl * (((bytes[q + l].toInt() and 0xFF) ushr shift) and 3) - ml
                    }
                    sc = bytes[scales + isIdx++].toInt() and 0xFF
                    dl = d * (sc and 0xF)
                    ml = min * (sc ushr 4)
                    for (l in 0 until 16) {
                        out[o++] = dl * (((bytes[q + l + 16].toInt() and 0xFF) ushr shift) and 3) - ml
                    }
                    shift += 2
                }
                q += 32
            }
        }
        return out
    }

    /**
     * `Q3_K`: 110 bytes for 256 elements — `hmask[32]`, `qs[64]`, `scales[12]`, then fp16 `d`
     * at the end. The 12 scale bytes unpack through ggml's `kmask` aux-word scheme into 16
     * signed 6-bit scales (offset by -32). A quant is `((qs >> shift) & 3) - (hmask bit ? 0 : 4)`,
     * with the hmask bit `m` advancing across the whole block. Bit-identical to llama.cpp's
     * `dequantize_row_q3_K`. (The previous decoder read `d` first, unpacked the scales as a flat
     * bit stream without the -32 offset, divided by 63 and added the high bit instead of
     * subtracting 4 — every value came out wrong; caught by `GgufQuantDequantReferenceTest`.)
     */
    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ3KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val bytesPerBlock = QK_K / 8 + QK_K / 4 + 12 + 2
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * QK_K)
        val kmask1 = 0x03030303
        val kmask2 = 0x0f0f0f0f
        val aux = IntArray(4)
        var o = 0
        for (i in 0 until blockCount) {
            val base = i * bytesPerBlock
            val hm = base
            val scalesBase = base + QK_K / 8 + QK_K / 4
            val dAll = halfToFloat(
                (bytes[scalesBase + 13].toInt() and 0xFF shl 8) or (bytes[scalesBase + 12].toInt() and 0xFF)
            )
            for (w in 0 until 3) {
                val p = scalesBase + 4 * w
                aux[w] = (bytes[p].toInt() and 0xFF) or
                    ((bytes[p + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[p + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[p + 3].toInt() and 0xFF) shl 24)
            }
            val tmp = aux[2]
            aux[2] = ((aux[0] ushr 4) and kmask2) or (((tmp ushr 4) and kmask1) shl 4)
            aux[3] = ((aux[1] ushr 4) and kmask2) or (((tmp ushr 6) and kmask1) shl 4)
            aux[0] = (aux[0] and kmask2) or (((tmp ushr 0) and kmask1) shl 4)
            aux[1] = (aux[1] and kmask2) or (((tmp ushr 2) and kmask1) shl 4)
            // The aux words read back as 16 *signed* bytes, exactly as C's `(const int8_t *) aux`.
            fun scaleAt(idx: Int): Int = ((aux[idx / 4] ushr (8 * (idx % 4))) and 0xFF).toByte().toInt()
            var isIdx = 0
            var m = 1
            var q = base + QK_K / 8
            for (half in 0 until 2) {
                var shift = 0
                for (j in 0 until 4) {
                    var dl = dAll * (scaleAt(isIdx++) - 32)
                    for (l in 0 until 16) {
                        val qv = ((bytes[q + l].toInt() and 0xFF) ushr shift) and 3
                        val sub = if ((bytes[hm + l].toInt() and m) != 0) 0 else 4
                        out[o++] = dl * (qv - sub)
                    }
                    dl = dAll * (scaleAt(isIdx++) - 32)
                    for (l in 0 until 16) {
                        val qv = ((bytes[q + l + 16].toInt() and 0xFF) ushr shift) and 3
                        val sub = if ((bytes[hm + l + 16].toInt() and m) != 0) 0 else 4
                        out[o++] = dl * (qv - sub)
                    }
                    shift += 2
                    m = m shl 1
                }
                q += 32
            }
        }
        return out
    }

    /**
     * ggml `get_scale_min_k4` reading straight from the packed buffer at
     * `scalesBase` — no per-block scratch copies (#782): the K-quant kernels
     * index into the source array directly so the only allocation a full-tensor
     * dequant makes is the destination `FloatArray`.
     */
    private fun getScaleMinK4(j: Int, bytes: ByteArray, scalesBase: Int): Pair<Int, Int> {
        return if (j < 4) {
            val sc = bytes[scalesBase + j].toInt() and 0x3F
            val m = bytes[scalesBase + j + 4].toInt() and 0x3F
            sc to m
        } else {
            val sc = ((bytes[scalesBase + j + 4].toInt() and 0x0F) or
                     (((bytes[scalesBase + j - 4].toInt() and 0xFF) shr 6) shl 4))
            val m = (((bytes[scalesBase + j + 4].toInt() and 0xFF) shr 4) or
                    (((bytes[scalesBase + j].toInt() and 0xFF) shr 6) shl 4))
            sc to m
        }
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ4KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 144
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
            val scalesBase = offset
            offset += 12
            val qsBase = offset
            offset += 128

            var qOffset = 0
            var scaleIdx = 0
            repeat(4) {
                val (sc1, m1) = getScaleMinK4(scaleIdx, bytes, scalesBase)
                val (sc2, m2) = getScaleMinK4(scaleIdx + 1, bytes, scalesBase)
                val d1 = d * sc1
                val min1 = dMin * m1
                val d2 = d * sc2
                val min2 = dMin * m2

                for (l in 0 until 32) {
                    val q = bytes[qsBase + qOffset + l].toInt() and 0x0F
                    out[outOff++] = d1 * q - min1
                }
                for (l in 0 until 32) {
                    val q = (bytes[qsBase + qOffset + l].toInt() and 0xFF) shr 4
                    out[outOff++] = d2 * q - min2
                }
                qOffset += 32
                scaleIdx += 2
            }
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ5KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 176
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
            val scalesBase = offset
            offset += 12
            val qhBase = offset
            offset += 32
            val qsBase = offset
            offset += 128

            // Per ggml-quants.c `dequantize_row_q5_K`: the 32-byte qh is indexed
            // by `l` (0..31, same as qs's per-group byte position), and a single
            // bit is selected per (outer-iter, low/high nibble). Different
            // (outer, nibble) pairs use different bits of the SAME qh[l] byte:
            //   outer 0: low→bit0, hi→bit1
            //   outer 1: low→bit2, hi→bit3
            //   outer 2: low→bit4, hi→bit5
            //   outer 3: low→bit6, hi→bit7
            // (Earlier this code used `qh[idx/8]` indexed by output position,
            // which only happened to equal qh[l] for blockCount=1; on real
            // multi-block tensors like Gemma 4 E2B's per_layer_token_embd
            // (Q5_K, 1.6 GB) every 5th bit was wrong, corrupting the PLE
            // residual stream across all 35 layers.)
            var qOffset = 0
            var scaleIdx = 0
            var outIdx = 0
            for (outer in 0 until 4) {
                val (sc1, m1) = getScaleMinK4(scaleIdx, bytes, scalesBase)
                val (sc2, m2) = getScaleMinK4(scaleIdx + 1, bytes, scalesBase)
                val d1 = d * sc1
                val min1 = dMin * m1
                val d2 = d * sc2
                val min2 = dMin * m2
                val bitLow = 2 * outer
                val bitHi = 2 * outer + 1

                for (l in 0 until 32) {
                    val qLow = bytes[qsBase + qOffset + l].toInt() and 0x0F
                    val qHigh = ((bytes[qhBase + l].toInt() and 0xFF) ushr bitLow) and 0x01
                    val q = qLow or (qHigh shl 4)
                    out[outOff + outIdx + l] = d1 * q - min1
                }
                for (l in 0 until 32) {
                    val qLow = (bytes[qsBase + qOffset + l].toInt() and 0xFF) ushr 4
                    val qHigh = ((bytes[qhBase + l].toInt() and 0xFF) ushr bitHi) and 0x01
                    val q = qLow or (qHigh shl 4)
                    out[outOff + outIdx + 32 + l] = d2 * q - min2
                }
                qOffset += 32
                scaleIdx += 2
                outIdx += 64
            }
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ6KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 210
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val qlStart = offset
            offset += 128
            val qhStart = offset
            offset += 64
            val scalesStart = offset
            offset += 16
            val d = halfToFloat(
                (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
            )
            offset += 2

            repeat(2) { half ->
                val qlBase = qlStart + half * 64
                val qhBase = qhStart + half * 32
                val scBase = scalesStart + half * 8

                for (l in 0 until 32) {
                    val isIdx = l / 16

                    val q1Low = bytes[qlBase + l].toInt() and 0x0F
                    val q1High = (bytes[qhBase + l].toInt() shr 0) and 0x03
                    val q1 = (q1Low or (q1High shl 4)) - 32

                    val q2Low = bytes[qlBase + l + 32].toInt() and 0x0F
                    val q2High = (bytes[qhBase + l].toInt() shr 2) and 0x03
                    val q2 = (q2Low or (q2High shl 4)) - 32

                    val q3Low = (bytes[qlBase + l].toInt() and 0xFF) shr 4
                    val q3High = (bytes[qhBase + l].toInt() shr 4) and 0x03
                    val q3 = (q3Low or (q3High shl 4)) - 32

                    val q4Low = (bytes[qlBase + l + 32].toInt() and 0xFF) shr 4
                    val q4High = (bytes[qhBase + l].toInt() shr 6) and 0x03
                    val q4 = (q4Low or (q4High shl 4)) - 32

                    val sc1 = bytes[scBase + isIdx + 0].toInt()
                    val sc2 = bytes[scBase + isIdx + 2].toInt()
                    val sc3 = bytes[scBase + isIdx + 4].toInt()
                    val sc4 = bytes[scBase + isIdx + 6].toInt()

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

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ8KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 292
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
            offset += 32 // skip bsums
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    /**
     * `TQ2_0` → floats through the shared reference codec (#1033).
     *
     * The layout — including the interleave, where one byte holds four elements **32 apart** — is
     * defined once by [sk.ainet.lang.memory.TernaryCodec], next to the `TensorEncoding.TQ2_0`
     * descriptor, so the loader and the ternary kernels cannot decode the same bytes differently.
     */
    private fun dequantTQ2_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blocks = bytes.size / sk.ainet.lang.tensor.storage.TensorEncoding.TQ2_0.BYTES_PER_BLOCK
        return sk.ainet.lang.memory.TernaryCodec.decodeTq2_0(bytes, blocks * 256)
    }

    /** `TQ1_0` → floats through the shared reference codec (#1033); see [dequantTQ2_0FromBytes]. */
    private fun dequantTQ1_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blocks = bytes.size / sk.ainet.lang.tensor.storage.TensorEncoding.TQ1_0.BYTES_PER_BLOCK
        return sk.ainet.lang.memory.TernaryCodec.decodeTq1_0(bytes, blocks * 256)
    }

    private fun typeName(value: Any?): String =
        value?.let { it::class.simpleName ?: it::class.toString() } ?: "null"
}
