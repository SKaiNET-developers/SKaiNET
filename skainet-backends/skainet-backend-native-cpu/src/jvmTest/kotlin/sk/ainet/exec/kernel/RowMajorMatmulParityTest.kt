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
            assertEquals(expected[o].toRawBits(), got[o].toRawBits(), "Q4_K row $o diverged: ${expected[o]} vs ${got[o]}")
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
            assertEquals(expected[o].toRawBits(), got[o].toRawBits(), "Q6_K row $o diverged: ${expected[o]} vs ${got[o]}")
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
}
