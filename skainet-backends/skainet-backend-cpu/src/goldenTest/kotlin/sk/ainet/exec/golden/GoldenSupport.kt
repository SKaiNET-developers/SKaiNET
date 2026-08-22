package sk.ainet.exec.golden

import kotlin.test.fail

/**
 * Support for the SKEEP-003 golden parity gate: deterministic inputs for every packed encoding
 * and a compact, bit-exact digest of the results.
 *
 * The digest is an FNV-1a 64-bit hash over the raw IEEE-754 bits of the produced floats (or the
 * produced bytes), plus the first four raw float bit patterns for a readable diff. A digest is
 * stable for the same bytes on every strict-binary32 target (JVM, Kotlin/Native); it changes the
 * moment a decoder or scalar kernel produces a different bit anywhere. To re-baseline after an
 * *intended* numeric change, run the tests and copy the "actual" value into [Goldens].
 */
internal object GoldenSupport {

    /** xorshift64* — small, allocation-free, identical on every target. */
    class Rng(seed: Long) {
        private var s: Long = if (seed == 0L) 0x9E3779B97F4A7C15uL.toLong() else seed
        fun nextLong(): Long {
            var x = s
            x = x xor (x ushr 12); x = x xor (x shl 25); x = x xor (x ushr 27)
            s = x
            return x * 0x2545F4914F6CDD1DuL.toLong()
        }
        fun nextByte(): Byte = (nextLong() ushr 56).toByte()
        /** Uniform in [0, 1) with 24 bits of randomness (exactly representable as Float). */
        fun nextFloat(): Float = ((nextLong() ushr 40).toInt() and 0xFFFFFF) / 16777216.0f
        /** Uniform in [-1, 1). */
        fun nextSigned(): Float = nextFloat() * 2f - 1f
    }

    /** Round-to-nearest-even FP32 → FP16 bits (same routine as the kernel parity tests). */
    fun half(v: Float): Int {
        val bits = v.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val expo = ((bits ushr 23) and 0xFF) - 127 + 15
        val mant = bits and 0x7FFFFF
        if (expo <= 0) return sign
        if (expo >= 31) return sign or 0x7C00
        return sign or (expo shl 10) or (mant ushr 13)
    }

    fun le16(b: ByteArray, off: Int, h: Int) {
        b[off] = (h and 0xFF).toByte(); b[off + 1] = ((h ushr 8) and 0xFF).toByte()
    }

    /** One packed GGML encoding: block geometry and a seeded block builder. */
    enum class Packed(val blockSize: Int, val bytesPerBlock: Int) {
        Q4_0(32, 18), Q5_0(32, 22), Q5_1(32, 24), Q8_0(32, 34), Q4_K(256, 144), Q5_K(256, 176), Q6_K(256, 210);

        /**
         * A random but *valid* block: every byte random, then the FP16 fields overwritten with sane
         * scales so no NaN/Inf can enter the arithmetic (NaN payloads are not portable).
         */
        fun block(rng: Rng): ByteArray {
            val b = ByteArray(bytesPerBlock) { rng.nextByte() }
            val d = rng.nextFloat() * 0.045f + 0.005f
            val dMin = rng.nextFloat() * 0.02f + 0.005f
            val m = rng.nextSigned() * 0.5f
            when (this) {
                Q4_0, Q5_0, Q8_0 -> le16(b, 0, half(d))
                Q5_1 -> { le16(b, 0, half(d)); le16(b, 2, half(m)) }
                Q4_K, Q5_K -> { le16(b, 0, half(d)); le16(b, 2, half(dMin)) }
                Q6_K -> le16(b, 208, half(d))
            }
            return b
        }
    }

    /** Seeded weight: [rows] output rows × [blocksPerRow] blocks, returned per (row, block). */
    fun weightBlocks(p: Packed, rows: Int, blocksPerRow: Int, seed: Long): Array<Array<ByteArray>> {
        val rng = Rng(seed)
        return Array(rows) { Array(blocksPerRow) { p.block(rng) } }
    }

    /** Canonical row-major bytes ([out, in] tensor on disk / in a TensorData). */
    fun rowMajor(blocks: Array<Array<ByteArray>>): ByteArray {
        val out = ArrayList<Byte>()
        for (row in blocks) for (blk in row) for (x in blk) out.add(x)
        return out.toByteArray()
    }

    /** Block-major bytes (the scalar kernels' input layout: `(blockIdx * outputDim + o)`). */
    fun blockMajor(blocks: Array<Array<ByteArray>>): ByteArray {
        val rows = blocks.size; val perRow = blocks[0].size
        val out = ArrayList<Byte>()
        for (bI in 0 until perRow) for (o in 0 until rows) for (x in blocks[o][bI]) out.add(x)
        return out.toByteArray()
    }

    fun floats(n: Int, seed: Long, scale: Float = 1f): FloatArray {
        val rng = Rng(seed)
        return FloatArray(n) { rng.nextSigned() * scale }
    }

    // --- digests ---

    private const val FNV_OFFSET = -3750763034362895579L // 0xcbf29ce484222325
    private const val FNV_PRIME = 1099511628211L

    private fun fnvByte(h: Long, b: Int): Long = (h xor (b and 0xFF).toLong()) * FNV_PRIME

    fun digest(values: FloatArray): String {
        var h = FNV_OFFSET
        for (v in values) {
            val bits = v.toRawBits()
            h = fnvByte(h, bits); h = fnvByte(h, bits ushr 8); h = fnvByte(h, bits ushr 16); h = fnvByte(h, bits ushr 24)
        }
        val head = (0 until minOf(4, values.size)).joinToString(",") { hex32(values[it].toRawBits()) }
        return "n=${values.size} fnv=${hex64(h)} head=$head"
    }

    fun digest(bytes: ByteArray): String {
        var h = FNV_OFFSET
        for (b in bytes) h = fnvByte(h, b.toInt())
        val head = (0 until minOf(8, bytes.size)).joinToString("") { hex8(bytes[it].toInt()) }
        return "n=${bytes.size} fnv=${hex64(h)} head=$head"
    }

    private fun hex8(v: Int): String = (v and 0xFF).toString(16).padStart(2, '0')
    private fun hex32(v: Int): String = v.toUInt().toString(16).padStart(8, '0')
    private fun hex64(v: Long): String = v.toULong().toString(16).padStart(16, '0')

    /** Assert [actual] equals the recorded golden for [name]; the failure message carries the actual digest. */
    fun check(name: String, actual: String) {
        val expected = Goldens.expected[name]
            ?: fail("No golden recorded for '$name'. Record it in Goldens.kt:\n    \"$name\" to \"$actual\",")
        if (expected != actual) {
            fail(
                "Golden mismatch for '$name' — the packed-encoding output is no longer bit-identical.\n" +
                "  expected: $expected\n  actual:   $actual\n" +
                "If this change is intended (new kernel semantics), re-baseline Goldens.kt in the same PR with the evidence."
            )
        }
    }
}
