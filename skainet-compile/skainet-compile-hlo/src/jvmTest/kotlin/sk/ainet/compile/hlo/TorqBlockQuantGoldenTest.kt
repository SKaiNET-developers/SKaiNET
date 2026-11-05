package sk.ainet.compile.hlo

import sk.ainet.compile.opt.passes.BlockQuantMatmulFusionPass
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
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Golden check against the Torq compiler: a Q8_0 and a Q4_0 `Linear` exported through
 * [BlockQuantMatmulFusionPass] must compile to `torq_hl.mixed_quant_matmul`, never to the generic
 * `torq_hl.matmul` fallback.
 *
 * Runs only when `SKAINET_TORQ_COMPILE` names a command that takes `torq-compile`'s arguments
 * (Torq 2.3.0 or later, with a linker for the aarch64 host code); otherwise it passes without
 * checking. Each case leaves `in.mlir`, `out.vmfb` and the dequantized weight `w.f32` (`[out, in]`)
 * in `build/torq-golden/<case>/` for a run on the board.
 */
class TorqBlockQuantGoldenTest {

    private val compiler: List<String>? = System.getenv("SKAINET_TORQ_COMPILE")?.takeIf { it.isNotBlank() }?.split(' ')

    @Test
    fun q8_0LinearCompilesToTheBlockQuantKernel() = check("q8_0", Q8_0Quantizer.quantize(weights(11), Shape(N, K)))

    @Test
    fun q4_0LinearWithNegativeScalesCompilesToTheBlockQuantKernel() = check("q4_0", Q4_0Quantizer.quantize(weights(12), Shape(N, K)))

    private fun check(name: String, weight: PackedBlockStorage) {
        val command = compiler ?: return
        val dir = File("build/torq-golden/$name").absoluteFile.apply { deleteRecursively(); mkdirs() }
        val input = File(dir, "in.mlir").apply { writeText(export(weight)) }
        File(dir, "w.f32").writeBytes(littleEndian(weight.toFloatArray()))
        val phases = File(dir, "phases")

        val process = ProcessBuilder(
            command + listOf(
                input.path,
                "--iree-input-type=stablehlo-torq",
                "--iree-hal-target-device=torq",
                "--torq-hw=SL2610",
                "--torq-fallback-f32-to-host",
                "--mlir-elide-elementsattrs-if-larger=8",
                "--dump-compilation-phases-to=${phases.path}",
                "-o", File(dir, "out.vmfb").path,
            )
        ).redirectErrorStream(true).redirectOutput(File(dir, "log.txt")).start()
        assertTrue(process.waitFor(10, TimeUnit.MINUTES), "torq-compile timed out")
        assertEquals(0, process.exitValue(), "torq-compile failed, see ${dir.path}/log.txt")

        val targets = assertNotNull(phases.listFiles()?.singleOrNull { it.name.endsWith(".executable-targets.mlir") }).readText()
        assertTrue("torq_hl.mixed_quant_matmul" in targets, "the block-quant kernel did not match, see ${phases.path}")
        assertTrue("torq_hl.matmul " !in targets, "part of the matmul fell back to the generic kernel")
    }

    private fun export(weight: PackedBlockStorage): String {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val x = ctx.fromFloatArray<FP32, Float>(Shape(1, K), FP32::class, FloatArray(K))
        @Suppress("UNCHECKED_CAST")
        val w = ctx.fromData(weight as TensorData<FP32, Float>, FP32::class)
        val xRef = ctx.session.refOf(x as Tensor<*, *>).id
        val (tape, _) = ctx.record { ctx.ops.matmulWeightTransposed(x, w) }
        val graph = assertNotNull(tape).toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(xRef),
            packedConstants = PackedConstantHandling.KEEP_PACKED,
        )
        return toStableHlo(BlockQuantMatmulFusionPass().apply(graph).graph).content
    }

    private fun weights(seed: Int): FloatArray {
        val r = Random(seed)
        return FloatArray(N * K) { r.nextFloat() * 2f - 1f }
    }

    private fun littleEndian(values: FloatArray): ByteArray =
        ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(values) }.array()

    private companion object {
        const val K = 256
        const val N = 96
    }
}
