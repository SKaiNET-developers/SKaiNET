package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32

/**
 * Ground-truth variant of the repro: unlike [NativeLazyTransposeAllZeroReproTest]
 * (which feeds the SAME flat byte array into both the "classic" and
 * "pre-transposed" constructions — which is mathematically guaranteed to
 * produce identical output no matter what, since `ops.transpose` is proven-
 * by-inspection to be a pure metadata relabel over the same bytes), this test
 * builds two bytewise-DIFFERENT packings of the SAME logical weight matrix:
 *
 * - "canonical" bytes: true row-major — for output row `o`, its
 *   `blocksPerInputDim` blocks are stored contiguously, then row `o+1`. This
 *   is what a loader constructing a `[outputDim, inputDim]`-shaped weight
 *   tensor "the way GGUF stores a row-major matrix" would naturally produce,
 *   and it is what `PackedBlockStorage.toFloatArray()` (block-sequential
 *   dequant) reconstructs as ground truth — reshaping its output as
 *   `[outputDim][inputDim]` gives the authoritative W[o][i] used nowhere near
 *   the matmul kernel under test.
 * - "kernel-native" bytes: input-block-major — `(blockIdx * outputDim + o)` —
 *   the physical order every native matmul kernel here actually assumes
 *   (see the q5_0_matmul.c / q4_0_matmul.c header comments). Same logical
 *   blocks, same per-(o,blockIdx) content, just placed at different flat
 *   offsets.
 *
 * `ops.transpose` claims a packed weight can flow from a `[outputDim,
 * inputDim]`-shaped tensor to `[inputDim, outputDim]` for free — same bytes,
 * just a shape relabel — because "the matmul kernels index the packed bytes
 * input-block-major from the post-swap shape" (`DefaultCpuOps.transpose`
 * doc comment). That claim is only true if the bytes were ALREADY
 * kernel-native (input-block-major) before the swap. If a weight is
 * genuinely row-major (canonical bytes, `blocksPerInputDim > 1`), the lazy
 * transpose does NOT reorder anything — so `ops.matmul(x, ops.transpose(w))`
 * on a canonically-packed weight hands the kernel bytes in the WRONG
 * physical order, silently, for every packed format with more than one
 * block per row.
 */
class NativeLazyTransposeGroundTruthReproTest {

    @BeforeTest
    fun forceNativeProviderOnly() {
        KernelRegistry.clearForTesting()
        KernelRegistry.register(NativeKernelProvider)
        assertTrue(NativeKernelProvider.isAvailable(), "native provider must be available for this repro")
    }

    @AfterTest
    fun reset() = KernelRegistry.clearForTesting()

    private fun le16(b: ByteArray, o: Int, h: Int) {
        b[o] = (h and 0xFF).toByte(); b[o + 1] = ((h ushr 8) and 0xFF).toByte()
    }

    /** One synthetic Q5_0/Q4_0/Q8_0/Q5_1 block's bytes, deterministic per (o, blockIdx). */
    private fun q5_0Block(o: Int, blockIdx: Int, seed: Int): ByteArray {
        val rng = Random(seed * 92821 + o * 977 + blockIdx)
        val b = ByteArray(22)
        le16(b, 0, 0x2C00 + rng.nextInt(0x80)) // small finite positive half
        for (k in 2 until 22) b[k] = rng.nextInt(256).toByte()
        return b
    }

    private fun q5_1Block(o: Int, blockIdx: Int, seed: Int): ByteArray {
        val rng = Random(seed * 92821 + o * 977 + blockIdx)
        val b = ByteArray(24)
        le16(b, 0, 0x2C00 + rng.nextInt(0x80)) // d
        le16(b, 2, 0x2400 + rng.nextInt(0x80)) // m
        for (k in 4 until 24) b[k] = rng.nextInt(256).toByte()
        return b
    }

    private fun q4_0Block(o: Int, blockIdx: Int, seed: Int): ByteArray {
        val rng = Random(seed * 92821 + o * 977 + blockIdx)
        val b = ByteArray(18)
        le16(b, 0, 0x2C00 + rng.nextInt(0x80))
        for (k in 2 until 18) b[k] = rng.nextInt(256).toByte()
        return b
    }

    private fun q8_0Block(o: Int, blockIdx: Int, seed: Int): ByteArray {
        val rng = Random(seed * 92821 + o * 977 + blockIdx)
        val b = ByteArray(34)
        le16(b, 0, 0x2C00 + rng.nextInt(0x80))
        for (k in 2 until 34) b[k] = rng.nextInt(256).toByte()
        return b
    }

    /** d @0, dMin @2 super-block scale header; ground truth only needs finite bytes. */
    private fun q4_kBlock(o: Int, blockIdx: Int, seed: Int): ByteArray {
        val rng = Random(seed * 92821 + o * 977 + blockIdx)
        val b = ByteArray(144)
        le16(b, 0, 0x3400 + rng.nextInt(0x80))
        le16(b, 2, 0x2C00 + rng.nextInt(0x80))
        for (k in 4 until 144) b[k] = rng.nextInt(256).toByte()
        return b
    }

    private fun q5_kBlock(o: Int, blockIdx: Int, seed: Int): ByteArray {
        val rng = Random(seed * 92821 + o * 977 + blockIdx)
        val b = ByteArray(176)
        le16(b, 0, 0x3400 + rng.nextInt(0x80))
        le16(b, 2, 0x2C00 + rng.nextInt(0x80))
        for (k in 4 until 176) b[k] = rng.nextInt(256).toByte()
        return b
    }

    /** Q6_K: `d` is the last two bytes of the 210-byte block. */
    private fun q6_kBlock(o: Int, blockIdx: Int, seed: Int): ByteArray {
        val rng = Random(seed * 92821 + o * 977 + blockIdx)
        val b = ByteArray(210)
        for (k in 0 until 208) b[k] = rng.nextInt(256).toByte()
        le16(b, 208, 0x3400 + rng.nextInt(0x80))
        return b
    }

    /**
     * Builds both the "canonical" (true row-major, output-major) and
     * "kernel-native" (input-block-major) flat packings from the SAME set of
     * per-(o,blockIdx) block contents.
     */
    private fun buildPackings(
        outputDim: Int,
        blocksPerInputDim: Int,
        bytesPerBlock: Int,
        blockAt: (o: Int, blockIdx: Int) -> ByteArray,
    ): Pair<ByteArray, ByteArray> {
        val canonical = ByteArray(outputDim * blocksPerInputDim * bytesPerBlock)
        val kernelNative = ByteArray(outputDim * blocksPerInputDim * bytesPerBlock)
        for (o in 0 until outputDim) {
            for (bI in 0 until blocksPerInputDim) {
                val block = blockAt(o, bI)
                block.copyInto(canonical, (o * blocksPerInputDim + bI) * bytesPerBlock)
                block.copyInto(kernelNative, (bI * outputDim + o) * bytesPerBlock)
            }
        }
        return canonical to kernelNative
    }

    private fun isAllZero(a: FloatArray): Boolean = a.all { it == 0f }

    private fun runGroundTruth(
        name: String,
        blockElems: Int,
        bytesPerBlock: Int,
        inputDim: Int,
        outputDim: Int,
        seed: Int,
        build: (Shape, ByteArray) -> TensorData<FP32, Float>,
        blockAt: (o: Int, blockIdx: Int) -> ByteArray,
    ) {
        val blocksPerInputDim = inputDim / blockElems
        val (canonicalBytes, kernelNativeBytes) = buildPackings(outputDim, blocksPerInputDim, bytesPerBlock, blockAt)

        // Ground truth: block-sequential dequant of the CANONICAL bytes reconstructs
        // W[o][i] row-major, independent of the matmul kernel under test.
        val wStorage = build(Shape(outputDim, inputDim), canonicalBytes) as PackedBlockStorage
        val wFlat = wStorage.toFloatArray() // row-major [outputDim][inputDim], length outputDim*inputDim
        val rng = Random(seed + 1000)
        val xf = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val yGroundTruth = FloatArray(outputDim) { o ->
            var s = 0f
            for (i in 0 until inputDim) s += xf[i] * wFlat[o * inputDim + i]
            s
        }

        val ctxClassic = DirectCpuExecutionContext()
        val wClassic = ctxClassic.fromData(build(Shape(outputDim, inputDim), canonicalBytes), FP32::class)
        val xClassic = ctxClassic.fromFloatArray<FP32, Float>(Shape(1, inputDim), FP32::class, xf)
        val yClassic = ctxClassic.ops.matmul(xClassic, ctxClassic.ops.transpose(wClassic)).data.copyToFloatArray()

        val ctxPre = DirectCpuExecutionContext()
        val wPre = ctxPre.fromData(build(Shape(inputDim, outputDim), kernelNativeBytes), FP32::class)
        val xPre = ctxPre.fromFloatArray<FP32, Float>(Shape(1, inputDim), FP32::class, xf)
        val yPre = ctxPre.ops.matmul(xPre, wPre).data.copyToFloatArray()

        // Tolerance relative to the output vector's overall scale (max |value|),
        // not each element individually: outputs near a sign change have tiny
        // |groundTruth[i]| where even small absolute FMA/summation-order noise
        // balloons into a huge *per-element* relative error despite being
        // consistent, small noise across the whole vector. Ground truth magnitudes
        // range from O(1) (32-elem blocks) to O(1e3) (256-elem K-quant
        // super-blocks summing 512 pseudo-random, largely-cancelling terms), so a
        // fixed absolute tolerance would either reject correct large-scale results
        // or accept near-zero-scale noise. A real dispatch bug (as originally
        // observed) produces errors that are large a fraction of the vector's own
        // scale (wrong sign, wrong magnitude on most/all outputs) — 1% of the
        // vector's max magnitude comfortably separates "FMA reordering noise" from
        // "reading the wrong bytes".
        val scale = maxOf(yGroundTruth.maxOf { abs(it) }, 1e-6f)
        fun matchesGroundTruth(y: FloatArray): Boolean = y.indices.all { i ->
            abs(y[i] - yGroundTruth[i]) <= 1e-2f * scale
        }
        val classicMatchesGroundTruth = matchesGroundTruth(yClassic)
        val preMatchesGroundTruth = matchesGroundTruth(yPre)

        println(
            "[$name] blocksPerInputDim=$blocksPerInputDim " +
                "classicAllZero=${isAllZero(yClassic)} classicMatchesGroundTruth=$classicMatchesGroundTruth " +
                "preAllZero=${isAllZero(yPre)} preMatchesGroundTruth=$preMatchesGroundTruth " +
                "groundTruth[0..3]=${yGroundTruth.take(4)} classic[0..3]=${yClassic.take(4)}",
        )

        // Regression contract (fixed): `ops.matmul(x, ops.transpose(w))` on a
        // canonically-packed weight — the "classic" path `linearProject` uses —
        // must match the SAME independent ground truth the "pre-transposed"
        // (skip-transpose, kernel-native-bytes) workaround already matched.
        // Before the fix this failed for every format with blocksPerInputDim > 1
        // (all of them here except the dedicated single-block control case).
        assertTrue(!isAllZero(yPre), "$name: pre-transposed native output must be non-zero")
        assertTrue(preMatchesGroundTruth, "$name: pre-transposed native output must match ground truth")
        assertTrue(!isAllZero(yClassic), "$name: classic (lazy-transpose) native output must be non-zero")
        assertTrue(
            classicMatchesGroundTruth,
            "$name: classic (lazy-transpose) native output must match ground truth " +
                "(SKaiNET-transformers#307 regression — transpose must physically reorder packed blocks)",
        )
    }

    @Test
    fun q5_0_ground_truth_multi_block_per_row() = runGroundTruth(
        "Q5_0", blockElems = 32, bytesPerBlock = 22, inputDim = 256, outputDim = 16, seed = 201,
        build = { s, b -> Q5_0BlockTensorData(s, b) as TensorData<FP32, Float> },
        blockAt = { o, bI -> q5_0Block(o, bI, 201) },
    )

    @Test
    fun q5_1_ground_truth_multi_block_per_row() = runGroundTruth(
        "Q5_1", blockElems = 32, bytesPerBlock = 24, inputDim = 256, outputDim = 16, seed = 202,
        build = { s, b -> Q5_1BlockTensorData(s, b) as TensorData<FP32, Float> },
        blockAt = { o, bI -> q5_1Block(o, bI, 202) },
    )

    @Test
    fun q4_0_ground_truth_multi_block_per_row() = runGroundTruth(
        "Q4_0", blockElems = 32, bytesPerBlock = 18, inputDim = 256, outputDim = 16, seed = 203,
        build = { s, b -> Q4_0BlockTensorData(s, b) as TensorData<FP32, Float> },
        blockAt = { o, bI -> q4_0Block(o, bI, 203) },
    )

    @Test
    fun q8_0_ground_truth_multi_block_per_row() = runGroundTruth(
        "Q8_0", blockElems = 32, bytesPerBlock = 34, inputDim = 256, outputDim = 16, seed = 204,
        build = { s, b -> Q8_0BlockTensorData(s, b) as TensorData<FP32, Float> },
        blockAt = { o, bI -> q8_0Block(o, bI, 204) },
    )

    @Test
    fun q4_k_ground_truth_multi_block_per_row() = runGroundTruth(
        "Q4_K", blockElems = 256, bytesPerBlock = 144, inputDim = 512, outputDim = 12, seed = 206,
        build = { s, b -> Q4_KBlockTensorData(s, b) as TensorData<FP32, Float> },
        blockAt = { o, bI -> q4_kBlock(o, bI, 206) },
    )

    @Test
    fun q5_k_ground_truth_multi_block_per_row() = runGroundTruth(
        "Q5_K", blockElems = 256, bytesPerBlock = 176, inputDim = 512, outputDim = 12, seed = 207,
        build = { s, b -> Q5_KBlockTensorData(s, b) as TensorData<FP32, Float> },
        blockAt = { o, bI -> q5_kBlock(o, bI, 207) },
    )

    @Test
    fun q6_k_ground_truth_multi_block_per_row() = runGroundTruth(
        "Q6_K", blockElems = 256, bytesPerBlock = 210, inputDim = 512, outputDim = 12, seed = 208,
        build = { s, b -> Q6_KBlockTensorData(s, b) as TensorData<FP32, Float> },
        blockAt = { o, bI -> q6_kBlock(o, bI, 208) },
    )

    // Single-block-per-row control: blocksPerInputDim == 1 collapses the
    // "input-block-major" and "output-major" orderings into the SAME physical
    // layout (block index reduces to `o` either way), so this is expected to
    // pass even if the general (multi-block-per-row) case is broken above.
    @Test
    fun q5_0_ground_truth_single_block_per_row_control() = runGroundTruth(
        "Q5_0-singleblock", blockElems = 32, bytesPerBlock = 22, inputDim = 32, outputDim = 16, seed = 205,
        build = { s, b -> Q5_0BlockTensorData(s, b) as TensorData<FP32, Float> },
        blockAt = { o, bI -> q5_0Block(o, bI, 205) },
    )
}
