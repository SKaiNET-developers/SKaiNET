package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * #1124: packed matmul must be right when **nothing** is registered.
 *
 * This configuration — the common `DefaultCpuOps` with an empty [KernelRegistry] — was the one
 * combination no test in the tree covered. The JVM tests always resolve to `DefaultCpuOpsJvm`,
 * whose override is correct; the native, JS and Wasm tests do use the common implementation, but
 * their platform factories register `ScalarKernelProvider` first. So every packed-matmul
 * correctness test, including the ones added for #973, #1096 and #1108, validated a configuration
 * that was not the broken one.
 *
 * The fallback exists precisely for when nothing else is available, which is exactly when it was
 * wrong — and wrong silently, by 4× the magnitude of the answer.
 */
class PackedMatmulEmptyRegistryTest {

    private val ctx = DirectCpuExecutionContext()

    @BeforeTest
    fun emptyTheRegistry() {
        // Build the context first: platform factories register ScalarKernelProvider on
        // construction, and this test is about what happens with nothing registered.
        ctx.ops
        KernelRegistry.clearForTesting()
        KernelDispatch.clearForTesting()
    }

    @AfterTest
    fun cleanup() {
        KernelRegistry.clearForTesting()
        KernelDispatch.clearForTesting()
    }

    /** Blocks whose codes differ per block, with a valid fp16 scale of 1.0 (and min 0.0 for Q5_1). */
    private fun bytes(name: String, blocks: Int, bytesPerBlock: Int): ByteArray {
        val out = ByteArray(blocks * bytesPerBlock)
        var seed = 7
        for (b in 0 until blocks) {
            val base = b * bytesPerBlock
            out[base] = 0x00; out[base + 1] = 0x3C
            if (name == "Q5_1") { out[base + 2] = 0x00; out[base + 3] = 0x00 }
            for (i in 4 until bytesPerBlock) {
                seed = seed * 1103515245 + 12345
                out[base + i] = ((seed ushr 16) % 9 - 4).toByte()
            }
        }
        return out
    }

    @Test
    fun `packed matmul agrees with the weight's own decoder when no kernel is registered`() {
        val cases: List<Triple<String, Int, (Shape, ByteArray) -> TensorData<FP32, Float>>> = listOf(
            Triple("Q8_0", 34) { s, b -> Q8_0BlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q4_0", 18) { s, b -> Q4_0BlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q5_0", 22) { s, b -> Q5_0BlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q5_1", 24) { s, b -> Q5_1BlockTensorData(s, b) as TensorData<FP32, Float> },
        )
        // Three blocks per row, so canonical and kernel-feed order differ (#968), and an output
        // dimension that is a whole number of blocks so the relayout is row-block-aligned.
        for ((name, bytesPerBlock, build) in cases) {
            for ((rows, cols) in listOf(32 to 96, 64 to 96, 32 to 128)) {
                val w: Tensor<FP32, Float> = ctx.fromData(
                    build(Shape(rows, cols), bytes(name, rows * (cols / 32), bytesPerBlock)), FP32::class,
                )
                val xs = FloatArray(cols) { (it % 13) * 0.0625f }
                val x = ctx.fromFloatArray<FP32, Float>(Shape(1, cols), FP32::class, xs)

                val decoded = (w.data as PackedBlockStorage).toFloatArray()
                val expected = FloatArray(rows) { o ->
                    var acc = 0.0
                    for (i in 0 until cols) acc += xs[i].toDouble() * decoded[o * cols + i]
                    acc.toFloat()
                }
                val actual = x.matmulWeightTransposed(w).data.copyToFloatArray()

                for (o in 0 until rows) {
                    assertTrue(
                        abs(expected[o] - actual[o]) <= 1e-3f * maxOf(1.0f, abs(expected[o])),
                        "$name [$rows x $cols] output[$o]: expected ${expected[o]}, got ${actual[o]}",
                    )
                }
            }
        }
    }
}
