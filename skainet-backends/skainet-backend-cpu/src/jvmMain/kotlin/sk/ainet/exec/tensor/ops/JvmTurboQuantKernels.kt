package sk.ainet.exec.tensor.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import sk.ainet.lang.tensor.ops.turboquant.BitPacker
import sk.ainet.lang.tensor.ops.turboquant.QuantizedVector
import sk.ainet.lang.tensor.ops.turboquant.ScalarQuantizer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/**
 * JVM SIMD-optimized kernels for TurboQuant operations.
 *
 * Uses the Java Vector API (jdk.incubator.vector) for CPU SIMD acceleration
 * of TurboQuant encode/decode paths. Falls back to scalar code for
 * non-aligned tails.
 *
 * These kernels optimize the hot paths:
 * - Per-group abs-max computation (for scale calculation)
 * - Vectorized quantization (float → code)
 * - Vectorized dequantization (code → float)
 * - Walsh-Hadamard transform butterfly stages
 *
 * Usage: Called by the CPU backend when TurboQuant-encoded K/V is detected
 * in the attention path.
 */
public object JvmTurboQuantKernels {

    private val FLOAT_SPECIES: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED
    private val floatStep: Int = FLOAT_SPECIES.length()

    // ========== Vectorized abs-max (for scale computation) ==========

    /**
     * Find the maximum absolute value in a float array segment.
     * SIMD-accelerated with scalar tail.
     */
    public fun absMax(data: FloatArray, offset: Int, length: Int): Float {
        var maxVec = FloatVector.zero(FLOAT_SPECIES)
        val end = offset + length
        val loopBound = FLOAT_SPECIES.loopBound(length) + offset
        var i = offset

        // Vectorized loop
        while (i < loopBound) {
            val v = FloatVector.fromArray(FLOAT_SPECIES, data, i)
            maxVec = maxVec.max(v.abs())
            i += floatStep
        }

        // Reduce vector to scalar
        var result = maxVec.reduceLanes(VectorOperators.MAX)

        // Scalar tail
        while (i < end) {
            result = max(result, abs(data[i]))
            i++
        }
        return result
    }

    // ========== Vectorized quantization ==========

    /**
     * SIMD-accelerated scalar quantization with per-group scales.
     *
     * Replaces [ScalarQuantizer.quantize] for the hot path.
     */
    public fun quantize(input: FloatArray, bits: Int): QuantizedVector {
        val maxCode = (1 shl (bits - 1)) - 1
        val groupSize = ScalarQuantizer.GROUP_SIZE
        val numGroups = (input.size + groupSize - 1) / groupSize
        val scales = FloatArray(numGroups)
        val codes = ByteArray(input.size)

        for (g in 0 until numGroups) {
            val start = g * groupSize
            val end = min(start + groupSize, input.size)
            val groupLen = end - start

            // SIMD abs-max
            val absMax = absMax(input, start, groupLen)
            val scale = if (absMax > 0f) absMax / maxCode else 0f
            scales[g] = scale

            if (scale > 0f) {
                val invScale = 1f / scale
                val invScaleVec = FloatVector.broadcast(FLOAT_SPECIES, invScale)
                val maxCodeF = maxCode.toFloat()
                val minCodeF = -maxCode.toFloat()
                val maxVec = FloatVector.broadcast(FLOAT_SPECIES, maxCodeF)
                val minVec = FloatVector.broadcast(FLOAT_SPECIES, minCodeF)

                val loopBound = FLOAT_SPECIES.loopBound(groupLen) + start
                var i = start

                // Vectorized quantize
                while (i < loopBound) {
                    val v = FloatVector.fromArray(FLOAT_SPECIES, input, i)
                    val scaled = v.mul(invScaleVec)
                    // Clamp to [-maxCode, maxCode]
                    val clamped = scaled.min(maxVec).max(minVec)
                    // Convert to int codes (round)
                    for (j in 0 until floatStep) {
                        codes[i + j] = round(clamped.lane(j)).toInt().toByte()
                    }
                    i += floatStep
                }

                // Scalar tail
                while (i < end) {
                    val q = round(input[i] * invScale).toInt()
                    codes[i] = q.coerceIn(-maxCode, maxCode).toByte()
                    i++
                }
            }
        }

        return QuantizedVector(codes, scales, bits)
    }

    // ========== Vectorized dequantization ==========

    /**
     * SIMD-accelerated dequantization.
     *
     * Replaces [ScalarQuantizer.dequantize] for the hot path.
     */
    public fun dequantize(codes: ByteArray, scales: FloatArray, output: FloatArray, offset: Int = 0) {
        val groupSize = ScalarQuantizer.GROUP_SIZE

        for (g in scales.indices) {
            val start = g * groupSize
            val end = min(start + groupSize, codes.size)
            val groupLen = end - start
            val scale = scales[g]
            val scaleVec = FloatVector.broadcast(FLOAT_SPECIES, scale)

            val loopBound = FLOAT_SPECIES.loopBound(groupLen) + start
            var i = start

            // Vectorized dequant: output = code * scale
            while (i < loopBound) {
                // Load codes as floats
                val floats = FloatArray(floatStep)
                for (j in 0 until floatStep) {
                    floats[j] = codes[i + j].toFloat()
                }
                val codeVec = FloatVector.fromArray(FLOAT_SPECIES, floats, 0)
                val result = codeVec.mul(scaleVec)
                result.intoArray(output, offset + i)
                i += floatStep
            }

            // Scalar tail
            while (i < end) {
                output[offset + i] = codes[i].toFloat() * scale
                i++
            }
        }
    }

    // ========== Vectorized Walsh-Hadamard butterfly ==========

    /**
     * SIMD-accelerated Walsh-Hadamard transform butterfly stage.
     *
     * Each butterfly stage computes: (a, b) → (a+b, a-b) for pairs
     * separated by stride `h`. The SIMD version processes multiple
     * pairs simultaneously.
     */
    public fun walshHadamardButterfly(data: FloatArray, h: Int, len: Int) {
        var i = 0
        while (i < len) {
            var j = i
            val jEnd = i + h
            val loopBound = FLOAT_SPECIES.loopBound(h) + i

            // Vectorized butterfly
            while (j < loopBound) {
                val a = FloatVector.fromArray(FLOAT_SPECIES, data, j)
                val b = FloatVector.fromArray(FLOAT_SPECIES, data, j + h)
                a.add(b).intoArray(data, j)
                a.sub(b).intoArray(data, j + h)
                j += floatStep
            }

            // Scalar tail
            while (j < jEnd) {
                val x = data[j]
                val y = data[j + h]
                data[j] = x + y
                data[j + h] = x - y
                j++
            }

            i += h * 2
        }
    }
}
