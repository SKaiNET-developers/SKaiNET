package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.MemSegKernelProvider

/**
 * Parity tests for [NativeQ4KMemSegMatmulKernel] vs the heap variant
 * [NativeQ4KMatmulKernel]. Both paths invoke the same
 * `skainet_q4k_matmul` C symbol, so output must agree **exactly** —
 * the only difference is whether the weight bytes were staged through
 * an arena copy or read directly from a caller-owned segment.
 *
 * Bit-identical assertion (no tolerance) is the contract: any drift
 * here means the wrapper added arithmetic, which is a bug.
 *
 * Also asserts the SPI plumbing works end-to-end:
 *  - [NativeKernelProvider] reports itself as [MemSegKernelProvider]
 *    so the smart-cast at the call site succeeds.
 *  - The factory class hands out the provider via the same path
 *    `KernelServiceLoader` would use.
 */
class NativeQ4KMemSegMatmulKernelParityTest {

    private val blockSize = 256
    private val bytesPerBlock = 144

    @BeforeTest
    fun checkAvailable() {
        assertTrue(NativeQ4KMatmulKernel.isAvailable(), "Heap kernel must be available")
        assertTrue(NativeQ4KMemSegMatmulKernel.isAvailable(), "MemSeg kernel must be available")
    }

    private fun randomQ4KBytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x3C.toByte()
            bytes[base + 2] = 0x00.toByte()
            bytes[base + 3] = 0x3C.toByte()
        }
        return bytes
    }

    private fun assertBitIdentical(inputDim: Int, outputDim: Int, seed: Int) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomQ4KBytes(numBlocks, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val heapOut = FloatArray(outputDim)
        NativeQ4KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, heapOut, 0)

        val memSegOut = FloatArray(outputDim)
        Arena.ofConfined().use { arena ->
            val weightSeg = arena.allocate(packed.size.toLong(), 1L)
            MemorySegment.copy(packed, 0, weightSeg, ValueLayout.JAVA_BYTE, 0L, packed.size)
            NativeQ4KMemSegMatmulKernel.matmul(
                input, 0,
                weightSeg, 0L,
                inputDim, outputDim,
                memSegOut, 0,
            )
        }

        for (o in 0 until outputDim) {
            assertEquals(
                heapOut[o].toRawBits(),
                memSegOut[o].toRawBits(),
                "row $o diverged: heap=${heapOut[o]} memSeg=${memSegOut[o]}",
            )
        }
    }

    @Test
    fun bit_identical_single_block_single_row() {
        assertBitIdentical(inputDim = 256, outputDim = 1, seed = 42)
    }

    @Test
    fun bit_identical_single_block_multi_row() {
        assertBitIdentical(inputDim = 256, outputDim = 16, seed = 7)
    }

    @Test
    fun bit_identical_multi_block_multi_row() {
        assertBitIdentical(inputDim = 1024, outputDim = 64, seed = 123)
    }

    @Test
    fun bit_identical_llm_typical_shape() {
        assertBitIdentical(inputDim = 4096, outputDim = 64, seed = 999)
    }

    @Test
    fun honors_non_zero_weight_byte_offset() {
        // Same weights laid out at byte offset 257 inside a larger
        // segment — kernel must skip the leading bytes correctly.
        val inputDim = 256
        val outputDim = 4
        val seed = 17
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomQ4KBytes(numBlocks, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val heapOut = FloatArray(outputDim)
        NativeQ4KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, heapOut, 0)

        val memSegOut = FloatArray(outputDim)
        val leadingPadBytes = 257L
        Arena.ofConfined().use { arena ->
            val weightSeg = arena.allocate(packed.size + leadingPadBytes, 1L)
            MemorySegment.copy(packed, 0, weightSeg, ValueLayout.JAVA_BYTE, leadingPadBytes, packed.size)
            NativeQ4KMemSegMatmulKernel.matmul(
                input, 0,
                weightSeg, leadingPadBytes,
                inputDim, outputDim,
                memSegOut, 0,
            )
        }

        for (o in 0 until outputDim) {
            assertEquals(heapOut[o].toRawBits(), memSegOut[o].toRawBits(), "row $o offset path diverged")
        }
    }

    @Test
    fun rejects_undersized_weight_segment() {
        val inputDim = 256
        val outputDim = 4
        val numBlocks = (inputDim / blockSize) * outputDim
        val needed = numBlocks.toLong() * bytesPerBlock // 4 * 144 = 576
        val input = FloatArray(inputDim)
        val output = FloatArray(outputDim)

        Arena.ofConfined().use { arena ->
            val tooSmall = arena.allocate(needed - 1L, 1L)
            try {
                NativeQ4KMemSegMatmulKernel.matmul(
                    input, 0,
                    tooSmall, 0L,
                    inputDim, outputDim,
                    output, 0,
                )
                kotlin.test.fail("expected IllegalArgumentException for undersized segment")
            } catch (_: IllegalArgumentException) {
                // expected
            }
        }
    }

    @Test
    fun provider_smart_casts_to_MemSegKernelProvider() {
        val provider: Any = NativeKernelProvider
        assertTrue(provider is MemSegKernelProvider, "NativeKernelProvider must implement MemSegKernelProvider")
        assertNotNull(provider.matmulQ4KMemSeg())
    }

    @Test
    fun factory_smart_casts_to_MemSegKernelProvider() {
        val factory: Any = NativeKernelProviderFactory()
        assertTrue(factory is MemSegKernelProvider, "Factory must implement MemSegKernelProvider for ServiceLoader path")
        assertNotNull(factory.matmulQ4KMemSeg())
    }
}
