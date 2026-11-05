package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.BlockQuantWeight
import sk.ainet.compile.opt.passes.BlockQuantMatmulFusionPass
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_0Quantizer
import sk.ainet.lang.tensor.data.Q8_0Quantizer
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.trace.PackedConstantHandling
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * A Q8_0 / Q4_0 `Linear` traced with `KEEP_PACKED`, folded by [BlockQuantMatmulFusionPass] and
 * emitted as the dequantize-then-`batch_matmul` linalg chain — the form the Torq compiler matches
 * to its block-quantized matmul kernel (checked against `torq-compile` by `TorqBlockQuantGoldenTest`).
 */
class BlockQuantMatmulExportTest {

    private val k = 64
    private val n = 48

    private fun weights(seed: Int, size: Int = n * k): FloatArray {
        val r = Random(seed)
        return FloatArray(size) { r.nextFloat() * 2f - 1f }
    }

    /** Trace `x · Wᵀ` with [packed] as W, then build the graph keeping packed weights. */
    private fun traceLinear(packed: TensorData<*, *>, xShape: Shape = Shape(1, k), op: String = "linear"): ComputeGraph {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val x = ctx.fromFloatArray<FP32, Float>(xShape, FP32::class, FloatArray(xShape.volume))
        @Suppress("UNCHECKED_CAST")
        val w = ctx.fromData(packed as TensorData<FP32, Float>, FP32::class)
        val xRef = ctx.session.refOf(x as Tensor<*, *>).id
        val (tape, _) = ctx.record {
            if (op == "linear") ctx.ops.matmulWeightTransposed(x, w) else ctx.ops.add(x, w)
        }
        return assertNotNull(tape).toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(xRef),
            packedConstants = PackedConstantHandling.KEEP_PACKED,
        )
    }

    private fun fuseAndExport(graph: ComputeGraph): String {
        val result = BlockQuantMatmulFusionPass().apply(graph)
        assertTrue(result.changed, "the pass must fold the packed Linear: ${result.diagnostics}")
        return toStableHlo(result.graph).content
    }

    @Test
    fun q8_0LinearBecomesTheI8Chain() {
        val mlir = fuseAndExport(traceLinear(Q8_0Quantizer.quantize(weights(1), Shape(n, k))))

        assertContains(mlir, "tensor<2x32x48xi8>")
        assertContains(mlir, "tensor<2x1x48xbf16>")
        assertContains(mlir, "affine_map<(d0, d1, d2) -> (d0, 0, d2)>")
        assertContains(mlir, "arith.sitofp")
        assertContains(mlir, "tensor.collapse_shape")
        assertContains(mlir, "output_shape [1, 64, 48]")
        assertContains(mlir, "linalg.batch_matmul")
        // f32 graph around a bf16 kernel
        assertContains(mlir, "stablehlo.convert")
        assertContains(mlir, "-> tensor<1x1x64xbf16>")
        assertFalse("stablehlo.dot_general" in mlir, "the matmul must not also be emitted dense")
        assertFalse("arith.subf" in mlir, "Q8_0 scales are positive: no zero point")
        assertFalse("dense<0.0>" in mlir, "no zero-filled stand-in for the weight")
    }

    @Test
    fun q4_0LinearWithNegativeScalesGetsAZeroPoint() {
        // ggml Q4_0 sets d = max / -8, so about half the blocks have a negative scale.
        val mlir = fuseAndExport(traceLinear(Q4_0Quantizer.quantize(weights(2), Shape(n, k))))

        assertContains(mlir, "tensor<2x32x48xi4>")
        assertContains(mlir, "arith.subf")
        assertEquals(2, Regex("""affine_map<\(d0, d1, d2\) -> \(d0, 0, d2\)>""").findAll(mlir).count(), "scale and zero point are both per block")
    }

    @Test
    fun batchedActivationsKeepTheirRows() {
        val mlir = fuseAndExport(traceLinear(Q8_0Quantizer.quantize(weights(3), Shape(n, k)), xShape = Shape(4, k)))
        assertContains(mlir, "tensor<1x4x64xbf16>")
        assertContains(mlir, "-> tensor<4x48xf32>")
    }

    @Test
    fun regroupedQ8_0WeightDecodesToTheSameMatrix() = assertRegroupedMatches(Q8_0Quantizer.quantize(weights(4), Shape(n, k)), bits = 8)

    @Test
    fun regroupedQ4_0WeightDecodesToTheSameMatrix() = assertRegroupedMatches(Q4_0Quantizer.quantize(weights(5), Shape(n, k)), bits = 4)

    private fun assertRegroupedMatches(storage: PackedBlockStorage, bits: Int) {
        val w = BlockQuantWeight.of(storage)
        assertEquals(bits, w.bits)
        val expected = storage.toFloatArray()
        val nB = k / 32
        val lo = -(1 shl (bits - 1))
        val hi = (1 shl (bits - 1)) - 1
        for (o in 0 until n) for (b in 0 until nB) {
            val scale = bf16ToFloat(w.scales[b * n + o])
            assertTrue(scale > 0f, "scale ($o, $b) must be positive for the kernel, was $scale")
            val zp = w.zeroPoints?.let { bf16ToFloat(it[b * n + o]) } ?: 0f
            for (j in 0 until 32) {
                val c = w.codes[(b * 32 + j) * n + o].toInt()
                assertTrue(c in lo..hi, "code $c out of i$bits range")
                val want = expected[o * k + b * 32 + j]
                val got = (c - zp) * scale
                // only the scale is rounded (f16 → bf16, 8 significant bits)
                assertTrue(abs(got - want) <= abs(want) / 128f + 1e-7f, "($o, ${b * 32 + j}): $got vs $want")
            }
        }
    }

    @Test
    fun allZeroBlockGetsAScaleTheKernelAccepts() {
        val w = BlockQuantWeight.of(Q8_0Quantizer.quantize(FloatArray(2 * 32), Shape(2, 32)))
        assertTrue(w.scales.all { bf16ToFloat(it) == 1f })
        assertTrue(w.codes.all { it == 0.toByte() })
        assertNull(w.zeroPoints)
    }

    @Test
    fun packedWeightNobodyFoldedFailsTheExport() {
        val graph = traceLinear(Q8_0Quantizer.quantize(weights(6, 64), Shape(1, 64)), xShape = Shape(1, 64), op = "add")
        val result = BlockQuantMatmulFusionPass().apply(graph)
        assertFalse(result.changed)
        assertTrue(result.diagnostics.any { it.startsWith("BLOCK_QUANT_SKIPPED") }, "${result.diagnostics}")
        val e = assertFailsWith<HloConversionException> { toStableHlo(result.graph) }
        assertContains(e.message.orEmpty(), "BlockQuantMatmulFusionPass")
    }

    private fun bf16ToFloat(bits: Short): Float = Float.fromBits((bits.toInt() and 0xFFFF) shl 16)
}
