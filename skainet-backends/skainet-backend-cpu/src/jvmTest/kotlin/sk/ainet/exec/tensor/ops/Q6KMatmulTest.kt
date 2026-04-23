package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Numerical parity tests for the Q6_K matmul kernel in
 * [JvmQuantizedVectorKernels.matmulQ6_KVec] against a reference dequant
 * implementation inlined from
 * `skainet-io-gguf.DequantOps.dequantQ6KFromBytes`. Inlined rather than
 * depending on `skainet-io-gguf` in test scope — keeps the backend test
 * dependency graph minimal.
 *
 * Verifies two invariants:
 * 1. Block-level dequant: `matmulQ6_KVec(input, packed, dim, 1, …)`
 *    on a one-output-row tensor matches `dot(input, dequantQ6K(packed))`.
 * 2. Multi-row matmul: the kernel produces the same results across
 *    `outputDim` rows as element-wise FP32 matmul after dequant.
 *
 * Test fixtures use random Q6_K bytes (not real checkpoint data) since
 * we're checking numerical parity between two code paths, not
 * correctness of specific weights.
 */
class Q6KMatmulTest {

    private val blockSize = 256
    private val bytesPerBlock = 210

    private fun randomQ6KBytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        // Clamp f16 bits to a reasonable range so we don't synthesize
        // NaN/Inf scales that'd make the parity comparison meaningless.
        for (block in 0 until numBlocks) {
            val dOffset = block * bytesPerBlock + 208
            // Force a small positive f16: exponent 0x0F (bias 15 → 2^0), mantissa 0.
            // 0x3C00 = 1.0f16.
            bytes[dOffset] = 0x00.toByte()
            bytes[dOffset + 1] = 0x3C.toByte()
        }
        return bytes
    }

    /**
     * Reference matmul: dequant Q6_K to FP32, then element-wise dot each
     * output row with the input. Uses the authoritative
     * `DequantOps.dequantFromBytes`.
     */
    private fun referenceMatmul(
        input: FloatArray,
        packedRowMajor: ByteArray,  // laid out [outputDim, blocksPerInput * bytesPerBlock] — row-major
        inputDim: Int,
        outputDim: Int
    ): FloatArray {
        val blocksPerInput = inputDim / blockSize
        val rowBytes = blocksPerInput * bytesPerBlock
        val out = FloatArray(outputDim)
        for (o in 0 until outputDim) {
            val rowBytesSlice = packedRowMajor.copyOfRange(o * rowBytes, (o + 1) * rowBytes)
            val row = referenceDequantQ6K(rowBytesSlice, inputDim)
            var sum = 0f
            for (i in 0 until inputDim) {
                sum += input[i] * row[i]
            }
            out[o] = sum
        }
        return out
    }

    /** Inlined from skainet-io-gguf.DequantOps.dequantQ6KFromBytes. */
    private fun referenceDequantQ6K(bytes: ByteArray, nElems: Int): FloatArray {
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val ql = bytes.copyOfRange(offset, offset + 128); offset += 128
            val qh = bytes.copyOfRange(offset, offset + 64); offset += 64
            val scales = bytes.copyOfRange(offset, offset + 16); offset += 16
            val d = halfBitsToFloat(
                ((bytes[offset + 1].toInt() and 0xFF) shl 8) or (bytes[offset].toInt() and 0xFF)
            )
            offset += 2

            repeat(2) { half ->
                val qlBase = half * 64
                val qhBase = half * 32
                val scBase = half * 8
                for (l in 0 until 32) {
                    val isIdx = l / 16
                    val q1Low = ql[qlBase + l].toInt() and 0x0F
                    val q1High = (qh[qhBase + l].toInt() shr 0) and 0x03
                    val q1 = (q1Low or (q1High shl 4)) - 32
                    val q2Low = ql[qlBase + l + 32].toInt() and 0x0F
                    val q2High = (qh[qhBase + l].toInt() shr 2) and 0x03
                    val q2 = (q2Low or (q2High shl 4)) - 32
                    val q3Low = (ql[qlBase + l].toInt() and 0xFF) shr 4
                    val q3High = (qh[qhBase + l].toInt() shr 4) and 0x03
                    val q3 = (q3Low or (q3High shl 4)) - 32
                    val q4Low = (ql[qlBase + l + 32].toInt() and 0xFF) shr 4
                    val q4High = (qh[qhBase + l].toInt() shr 6) and 0x03
                    val q4 = (q4Low or (q4High shl 4)) - 32
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

    private fun halfBitsToFloat(h: Int): Float {
        val sign = (h and 0x8000) shl 16
        val exp = (h and 0x7C00) shr 10
        val mant = h and 0x03FF
        return when (exp) {
            0 -> {
                if (mant == 0) Float.fromBits(sign)
                else {
                    var m = mant; var e = -14
                    while ((m and 0x400) == 0) { m = m shl 1; e-- }
                    m = m and 0x3FF
                    Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
                }
            }
            31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }

    /**
     * Convert row-major `[outputDim, blocks, 210B]` layout to the
     * input-block-major `[blocks, outputDim, 210B]` layout that
     * `matmulQ6_KVec` expects. This is exactly
     * `GemmaMemSegConverter.relayoutQ4_KRowMajorToBlockMajor` but for
     * Q6_K's 210-byte blocks.
     */
    private fun relayoutRowMajorToBlockMajor(
        rowMajor: ByteArray,
        inputDim: Int,
        outputDim: Int
    ): ByteArray {
        val blocksPerRow = inputDim / blockSize
        val out = ByteArray(rowMajor.size)
        for (o in 0 until outputDim) {
            for (b in 0 until blocksPerRow) {
                val src = (o * blocksPerRow + b) * bytesPerBlock
                val dst = (b * outputDim + o) * bytesPerBlock
                rowMajor.copyInto(out, dst, src, src + bytesPerBlock)
            }
        }
        return out
    }

    @Test
    fun `matmulQ6_KVec single output row matches reference dequant dot`() {
        val inputDim = 256
        val outputDim = 1
        val packedRowMajor = randomQ6KBytes(outputDim * (inputDim / blockSize), seed = 42)
        val input = FloatArray(inputDim) { Random(1).nextFloat() - 0.5f }

        val kernelOut = FloatArray(outputDim)
        val packedBlockMajor = relayoutRowMajorToBlockMajor(packedRowMajor, inputDim, outputDim)
        JvmQuantizedVectorKernels.matmulQ6_KVec(
            input, packedBlockMajor, inputDim, outputDim, kernelOut, 0
        )
        val expected = referenceMatmul(input, packedRowMajor, inputDim, outputDim)

        val diff = abs(kernelOut[0] - expected[0])
        val rel = diff / (abs(expected[0]) + 1e-9f)
        assertTrue(
            rel < 1e-4f,
            "single-row Q6_K matmul diverged: kernel=${kernelOut[0]} ref=${expected[0]} diff=$diff rel=$rel"
        )
    }

    @Test
    fun `matmulQ6_KVec multi output multi block matches reference`() {
        val inputDim = 256 * 3   // 3 blocks
        val outputDim = 4
        val packedRowMajor = randomQ6KBytes(outputDim * (inputDim / blockSize), seed = 123)
        val input = FloatArray(inputDim) { Random(it).nextFloat() - 0.5f }

        val kernelOut = FloatArray(outputDim)
        val packedBlockMajor = relayoutRowMajorToBlockMajor(packedRowMajor, inputDim, outputDim)
        JvmQuantizedVectorKernels.matmulQ6_KVec(
            input, packedBlockMajor, inputDim, outputDim, kernelOut, 0
        )
        val expected = referenceMatmul(input, packedRowMajor, inputDim, outputDim)

        for (o in 0 until outputDim) {
            val diff = abs(kernelOut[o] - expected[o])
            val rel = diff / (abs(expected[o]) + 1e-9f)
            // 1e-4 relative tolerance: SIMD reduction-order vs scalar
            // reduction drifts a few ULPs per accumulation; on random
            // Q6_K bytes the scale arithmetic gives magnitudes ~1e4 so
            // absolute diff is ~1e-2.
            assertTrue(
                rel < 1e-4f,
                "Q6_K matmul row $o diverged: kernel=${kernelOut[o]} ref=${expected[o]} diff=$diff rel=$rel"
            )
        }
    }
}
