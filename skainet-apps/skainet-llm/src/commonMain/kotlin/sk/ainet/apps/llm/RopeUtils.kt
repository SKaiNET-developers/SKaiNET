package sk.ainet.apps.llm

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * Compute RoPE (Rotary Position Embedding) frequency for a given pair index and position.
 *
 * Formula: freq = pos / base^(2 * pair / dim)
 *
 * @param pair Pair index (0 to dim/2 - 1)
 * @param pos Sequence position
 * @param dim RoPE dimension (typically headSize)
 * @param base Base frequency (default 10000, Gemma global uses 1M)
 */
public fun ropeFrequency(pair: Int, pos: Int, dim: Int, base: Float = 10000f): Float {
    val exponent = (2f * pair) / dim
    return pos / base.pow(exponent)
}

/**
 * Compute RoPE cosine value for a given pair and position.
 */
public fun ropeCos(pair: Int, pos: Int, dim: Int, base: Float = 10000f): Float =
    cos(ropeFrequency(pair, pos, dim, base))

/**
 * Compute RoPE sine value for a given pair and position.
 */
public fun ropeSin(pair: Int, pos: Int, dim: Int, base: Float = 10000f): Float =
    sin(ropeFrequency(pair, pos, dim, base))

/**
 * Apply RoPE rotation in-place to a buffer of head vectors.
 *
 * Rotates pairs of elements (even, odd) using the RoPE formula:
 *   out[2i]   = in[2i] * cos(freq) - in[2i+1] * sin(freq)
 *   out[2i+1] = in[2i] * sin(freq) + in[2i+1] * cos(freq)
 *
 * Supports optional precomputed cos/sin tables for performance.
 * When [precomputedMatchBase] is provided, precomputed tables are only used
 * if [base] matches that value (useful for Gemma's dual-frequency RoPE).
 *
 * @param buf Buffer containing concatenated head vectors (mutated in-place)
 * @param nHeads Number of heads in the buffer
 * @param headSize Size of each head
 * @param ropeDim Number of dimensions to rotate (typically headSize)
 * @param pos Current sequence position
 * @param base RoPE base frequency
 * @param precomputedCos Optional precomputed cosine table [pos * ropeStride + pair]
 * @param precomputedSin Optional precomputed sine table [pos * ropeStride + pair]
 * @param ropeStride Stride in precomputed tables (typically headSize / 2)
 * @param precomputedMatchBase If set, only use precomputed tables when base equals this value
 */
public fun applyRopeRotation(
    buf: FloatArray,
    nHeads: Int,
    headSize: Int,
    ropeDim: Int,
    pos: Int,
    base: Float = 10000f,
    precomputedCos: FloatArray? = null,
    precomputedSin: FloatArray? = null,
    ropeStride: Int = ropeDim / 2,
    precomputedMatchBase: Float? = null
) {
    val usePrecomputed = precomputedCos != null && precomputedSin != null &&
            (precomputedMatchBase == null || base == precomputedMatchBase)

    for (h in 0 until nHeads) {
        val headOffset = h * headSize
        for (pair in 0 until ropeDim / 2) {
            val i = pair * 2
            val fcr: Float
            val fci: Float
            if (usePrecomputed) {
                fcr = precomputedCos!![pos * ropeStride + pair]
                fci = precomputedSin!![pos * ropeStride + pair]
            } else {
                fcr = ropeCos(pair, pos, ropeDim, base)
                fci = ropeSin(pair, pos, ropeDim, base)
            }
            val v0 = buf[headOffset + i]
            val v1 = buf[headOffset + i + 1]
            buf[headOffset + i] = v0 * fcr - v1 * fci
            buf[headOffset + i + 1] = v0 * fci + v1 * fcr
        }
    }
}
