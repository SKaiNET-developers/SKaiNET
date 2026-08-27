package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Parity tests for the row-major kernel variants `skainet_q4k_matmul_rm` /
 * `skainet_q6k_matmul_rm` (#1189) against the feed-order kernels they mirror.
 *
 * The same logical weight is laid out both ways — row-major `(o * blocksPerRow + b)` and
 * feed-order `(b * outputDim + o)` — and both kernels must produce **bit-identical** outputs:
 * per output row the accumulation order over blocks is the same, so any drift means the row
 * addressing (the only thing that differs) is wrong.
 */
class RowMajorMatmulParityTest {

    private companion object {
        const val BLOCK = 256
        const val Q4K_BPB = 144
        const val Q6K_BPB = 210
        const val Q5K_BPB = 176
        const val SMALL = 32
        const val Q80_BPB = 34
        const val Q40_BPB = 18
        const val Q50_BPB = 22
        const val Q51_BPB = 24
    }

    private fun bindRm(symbol: String): MethodHandle? {
        val lookup = NativeLibraryLoader.lookup() ?: return null
        val sym = lookup.find(symbol).orElse(null) ?: return null
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,   // input, input_offset
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,   // weight, weight_byte_offset
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,  // input_dim, output_dim
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,   // output, output_offset
        )
        return runCatching { Linker.nativeLinker().downcallHandle(sym, descriptor) }.getOrNull()
    }

    private val q4kRm: MethodHandle? by lazy { bindRm("skainet_q4k_matmul_rm") }
    private val q6kRm: MethodHandle? by lazy { bindRm("skainet_q6k_matmul_rm") }

    @BeforeTest
    fun checkAvailable() {
        assertTrue(NativeQ4KMatmulKernel.isAvailable(), "feed-order Q4_K kernel must be available")
        assertTrue(NativeQ6KMatmulKernel.isAvailable(), "feed-order Q6_K kernel must be available")
        assertNotNull(q4kRm, "skainet_q4k_matmul_rm must be bindable")
        assertNotNull(q6kRm, "skainet_q6k_matmul_rm must be bindable")
    }

    /** Random block bytes with the FP16 scale fields pinned to 1.0 so magnitudes stay sane. */
    private fun randomBlocks(numBlocks: Int, bytesPerBlock: Int, fp16At: IntArray, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            for (off in fp16At) {
                bytes[base + off] = 0x00
                bytes[base + off + 1] = 0x3C  // 1.0f as binary16
            }
        }
        return bytes
    }

    /** Permute row-major blocks `(o * bpr + b)` into feed order `(b * n + o)`. */
    private fun toFeedOrder(rowMajor: ByteArray, n: Int, bpr: Int, bpb: Int): ByteArray {
        val out = ByteArray(rowMajor.size)
        for (o in 0 until n) {
            for (b in 0 until bpr) {
                rowMajor.copyInto(
                    out,
                    destinationOffset = (b * n + o) * bpb,
                    startIndex = (o * bpr + b) * bpb,
                    endIndex = (o * bpr + b + 1) * bpb,
                )
            }
        }
        return out
    }

    private fun callRm(
        handle: MethodHandle,
        input: FloatArray,
        rowMajorWeight: ByteArray,
        weightByteOffset: Int,
        inputDim: Int,
        outputDim: Int,
    ): FloatArray {
        val out = FloatArray(outputDim)
        Arena.ofConfined().use { arena ->
            val inSeg = arena.allocate(inputDim.toLong() * 4, 4)
            MemorySegment.copy(input, 0, inSeg, ValueLayout.JAVA_FLOAT, 0L, inputDim)
            val wSeg = arena.allocate(rowMajorWeight.size.toLong(), 1)
            MemorySegment.copy(rowMajorWeight, 0, wSeg, ValueLayout.JAVA_BYTE, 0L, rowMajorWeight.size)
            val outSeg = arena.allocate(outputDim.toLong() * 4, 4)
            handle.invoke(inSeg, 0, wSeg, weightByteOffset, inputDim, outputDim, outSeg, 0)
            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0L, out, 0, outputDim)
        }
        return out
    }


    /**
     * Cross-ORDER comparisons are numerically-equivalent, not bit-exact: under -O3 -ffast-math
     * the compiler may contract/re-associate float accumulation differently between the two loop
     * shapes (q5_1 measured 2 ULP apart; the integer-dot formats happen to match exactly).
     * Threaded-vs-solo comparisons stay bit-exact — same function, same per-row order.
     */
    private fun assertClose(expected: Float, got: Float, label: String) {
        val tol = maxOf(1e-5f, 1e-5f * kotlin.math.abs(expected))
        assertTrue(kotlin.math.abs(expected - got) <= tol, "$label: $expected vs $got (tol $tol)")
    }

    private fun assertQ4kParity(inputDim: Int, outputDim: Int, seed: Int, pad: Int = 0) {
        val bpr = inputDim / BLOCK
        val rowMajor = randomBlocks(bpr * outputDim, Q4K_BPB, intArrayOf(0, 2), seed)
        val feed = toFeedOrder(rowMajor, outputDim, bpr, Q4K_BPB)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val expected = FloatArray(outputDim)
        NativeQ4KMatmulKernel.matmul(input, 0, feed, 0, inputDim, outputDim, expected, 0)

        val padded = if (pad == 0) rowMajor else ByteArray(pad) + rowMajor
        val got = callRm(q4kRm!!, input, padded, pad, inputDim, outputDim)
        for (o in 0 until outputDim) {
            assertClose(expected[o], got[o], "Q4_K row $o")
        }
    }

    private fun assertQ6kParity(inputDim: Int, outputDim: Int, seed: Int, pad: Int = 0) {
        val bpr = inputDim / BLOCK
        // Q6_K: d is the trailing FP16 at byte 208.
        val rowMajor = randomBlocks(bpr * outputDim, Q6K_BPB, intArrayOf(208), seed)
        val feed = toFeedOrder(rowMajor, outputDim, bpr, Q6K_BPB)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val expected = FloatArray(outputDim)
        NativeQ6KMatmulKernel.matmul(input, 0, feed, 0, inputDim, outputDim, expected, 0)

        val padded = if (pad == 0) rowMajor else ByteArray(pad) + rowMajor
        val got = callRm(q6kRm!!, input, padded, pad, inputDim, outputDim)
        for (o in 0 until outputDim) {
            assertClose(expected[o], got[o], "Q6_K row $o")
        }
    }

    @Test fun q4k_single_block_single_row() = assertQ4kParity(256, 1, seed = 42)
    @Test fun q4k_single_block_multi_row() = assertQ4kParity(256, 16, seed = 7)
    @Test fun q4k_multi_block_multi_row() = assertQ4kParity(1024, 64, seed = 123)
    @Test fun q4k_llm_typical_shape() = assertQ4kParity(1536, 48, seed = 999)
    @Test fun q4k_honors_weight_byte_offset() = assertQ4kParity(512, 8, seed = 17, pad = 257)

    @Test fun q6k_single_block_single_row() = assertQ6kParity(256, 1, seed = 41)
    @Test fun q6k_single_block_multi_row() = assertQ6kParity(256, 16, seed = 8)
    @Test fun q6k_multi_block_multi_row() = assertQ6kParity(1024, 64, seed = 321)
    @Test fun q6k_honors_weight_byte_offset() = assertQ6kParity(512, 8, seed = 18, pad = 129)

    // ---- #1195: outputDim >= 512 engages the row-partition threading. Two oracles: ----

    /** Threaded full-matrix call vs the feed-order kernel on permuted bytes (both threaded). */
    @Test fun q4k_threaded_parity_vs_feed_order() = assertQ4kParity(512, 1536, seed = 77)
    @Test fun q6k_threaded_parity_vs_feed_order() = assertQ6kParity(512, 1536, seed = 78)

    /**
     * Threaded full-matrix call vs 1536 independent single-row calls (each under the
     * threshold, so single-threaded) — pins the partition arithmetic itself: every row of a
     * threaded call must be bit-identical to that row computed alone.
     */
    @Test
    fun q4k_threaded_equals_per_row_calls() {
        val inputDim = 512
        val n = 1536
        val bpr = inputDim / BLOCK
        val rowMajor = randomBlocks(bpr * n, Q4K_BPB, intArrayOf(0, 2), seed = 91)
        val input = FloatArray(inputDim) { Random(91 + it).nextFloat() - 0.5f }

        val full = callRm(q4kRm!!, input, rowMajor, 0, inputDim, n)
        for (o in 0 until n step 97) {
            val single = callRm(q4kRm!!, input, rowMajor, o * bpr * Q4K_BPB, inputDim, 1)
            assertEquals(single[0].toRawBits(), full[o].toRawBits(), "Q4_K row $o: threaded diverged from solo")
        }
    }

    @Test
    fun q6k_threaded_equals_per_row_calls() {
        val inputDim = 512
        val n = 1536
        val bpr = inputDim / BLOCK
        val rowMajor = randomBlocks(bpr * n, Q6K_BPB, intArrayOf(208), seed = 92)
        val input = FloatArray(inputDim) { Random(92 + it).nextFloat() - 0.5f }

        val full = callRm(q6kRm!!, input, rowMajor, 0, inputDim, n)
        for (o in 0 until n step 97) {
            val single = callRm(q6kRm!!, input, rowMajor, o * bpr * Q6K_BPB, inputDim, 1)
            assertEquals(single[0].toRawBits(), full[o].toRawBits(), "Q6_K row $o: threaded diverged from solo")
        }
    }

    // ---- #1192: the four remaining GGML block formats, same two-oracle scheme ----

    private fun assertParity(
        rmSymbol: String,
        feed: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
        blockSize: Int, bpb: Int, fp16At: IntArray,
        inputDim: Int, outputDim: Int, seed: Int,
    ) {
        val handle = bindRm(rmSymbol)
        assertNotNull(handle, "$rmSymbol must be bindable")
        val bpr = inputDim / blockSize
        val rowMajor = randomBlocks(bpr * outputDim, bpb, fp16At, seed)
        val feedBytes = toFeedOrder(rowMajor, outputDim, bpr, bpb)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val expected = FloatArray(outputDim)
        feed(input, 0, feedBytes, 0, inputDim, outputDim, expected, 0)
        val got = callRm(handle, input, rowMajor, 0, inputDim, outputDim)
        for (o in 0 until outputDim) {
            assertClose(expected[o], got[o], "$rmSymbol row $o")
        }
    }

    @Test fun q5k_rm_parity() = assertParity(
        "skainet_q5k_matmul_rm", NativeQ5KMatmulKernel::matmul, BLOCK, Q5K_BPB, intArrayOf(0, 2), 512, 48, seed = 21)
    @Test fun q5k_rm_parity_threaded() = assertParity(
        "skainet_q5k_matmul_rm", NativeQ5KMatmulKernel::matmul, BLOCK, Q5K_BPB, intArrayOf(0, 2), 512, 1024, seed = 22)
    @Test fun q80_rm_parity() = assertParity(
        "skainet_q8_0_matmul_rm", NativeQ8_0MatmulKernel::matmul, SMALL, Q80_BPB, intArrayOf(0), 256, 48, seed = 23)
    @Test fun q80_rm_parity_threaded() = assertParity(
        "skainet_q8_0_matmul_rm", NativeQ8_0MatmulKernel::matmul, SMALL, Q80_BPB, intArrayOf(0), 256, 1024, seed = 24)
    @Test fun q40_rm_parity() = assertParity(
        "skainet_q4_0_matmul_rm", NativeQ4_0MatmulKernel::matmul, SMALL, Q40_BPB, intArrayOf(0), 256, 48, seed = 25)
    @Test fun q40_rm_parity_threaded() = assertParity(
        "skainet_q4_0_matmul_rm", NativeQ4_0MatmulKernel::matmul, SMALL, Q40_BPB, intArrayOf(0), 256, 1024, seed = 26)
    @Test fun q50_rm_parity() = assertParity(
        "skainet_q5_0_matmul_rm", NativeQ5_0MatmulKernel::matmul, SMALL, Q50_BPB, intArrayOf(0), 256, 48, seed = 27)
    @Test fun q50_rm_parity_threaded() = assertParity(
        "skainet_q5_0_matmul_rm", NativeQ5_0MatmulKernel::matmul, SMALL, Q50_BPB, intArrayOf(0), 256, 1024, seed = 28)
    @Test fun q51_rm_parity() = assertParity(
        "skainet_q5_1_matmul_rm", NativeQ5_1MatmulKernel::matmul, SMALL, Q51_BPB, intArrayOf(0, 2), 256, 48, seed = 29)
    @Test fun q51_rm_parity_threaded() = assertParity(
        "skainet_q5_1_matmul_rm", NativeQ5_1MatmulKernel::matmul, SMALL, Q51_BPB, intArrayOf(0, 2), 256, 1024, seed = 30)
}
