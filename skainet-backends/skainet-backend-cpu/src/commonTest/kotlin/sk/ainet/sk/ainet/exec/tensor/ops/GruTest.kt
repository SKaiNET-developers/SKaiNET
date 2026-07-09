package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.math.exp
import kotlin.math.tanh
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.Gru
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/**
 * Eager forward correctness for the GRU layer, checked against an independent
 * scalar reference (raw FloatArray loops) computing the same PyTorch r,z,n
 * recurrence. Exercises all three gates, the per-timestep hidden feedback
 * (weight_hh path) and the update blend.
 */
class GruTest {
    private val ctx = DirectCpuExecutionContext()

    private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

    // Independent reference: input [S*D] (batch=1), weights matmul-ready (row-major).
    private fun gruRef(
        x: FloatArray, seq: Int, d: Int, h: Int,
        wIh: FloatArray, wHh: FloatArray, bIh: FloatArray, bHh: FloatArray,
    ): FloatArray {
        val g = 3 * h
        val hidden = FloatArray(h)
        val out = FloatArray(seq * h)
        for (t in 0 until seq) {
            val gx = FloatArray(g) { k -> bIh[k] + (0 until d).sumOf { i -> (x[t * d + i] * wIh[i * g + k]).toDouble() }.toFloat() }
            val gh = FloatArray(g) { k -> bHh[k] + (0 until h).sumOf { j -> (hidden[j] * wHh[j * g + k]).toDouble() }.toFloat() }
            for (j in 0 until h) {
                val r = sigmoid(gx[j] + gh[j])
                val z = sigmoid(gx[h + j] + gh[h + j])
                val n = tanh((gx[2 * h + j] + r * gh[2 * h + j]).toDouble()).toFloat()
                hidden[j] = (1f - z) * n + z * hidden[j]
            }
            for (j in 0 until h) out[t * h + j] = hidden[j]
        }
        return out
    }

    @Test
    fun gru_forward_matches_reference() {
        val batch = 1; val seq = 2; val d = 2; val h = 2; val g = 3 * h
        // deterministic small weights/inputs (kept in sigmoid/tanh's sensitive range)
        val x = FloatArray(seq * d) { ((it % 5) - 2) * 0.3f }
        val wIh = FloatArray(d * g) { ((it % 7) - 3) * 0.1f }
        val wHh = FloatArray(h * g) { ((it % 5) - 2) * 0.15f }
        val bIh = FloatArray(g) { ((it % 3) - 1) * 0.2f }
        val bHh = FloatArray(g) { ((it % 4) - 2) * 0.1f }

        val gru = Gru<FP32, Float>(
            inputSize = d, hiddenSize = h, name = "gru",
            initWeightIh = ctx.fromFloatArray(Shape(d, g), FP32::class, wIh),
            initWeightHh = ctx.fromFloatArray(Shape(h, g), FP32::class, wHh),
            initBiasIh = ctx.fromFloatArray(Shape(g), FP32::class, bIh),
            initBiasHh = ctx.fromFloatArray(Shape(g), FP32::class, bHh),
        )

        val input = ctx.fromFloatArray<FP32, Float>(Shape(batch, seq, d), FP32::class, x)
        val out = gru.forward(input, ctx)
        assertEquals(Shape(batch, seq, h), out.shape)

        val expected = gruRef(x, seq, d, h, wIh, wHh, bIh, bHh)
        for (t in 0 until seq) for (j in 0 until h) {
            assertEquals(expected[t * h + j], out.data[0, t, j], 1e-5f)
        }
    }

    @Test
    fun gru_output_shape_is_batch_seq_hidden() {
        val batch = 2; val seq = 3; val d = 4; val h = 5; val g = 3 * h
        val gru = Gru<FP32, Float>(
            inputSize = d, hiddenSize = h, name = "gru",
            initWeightIh = ctx.fromFloatArray(Shape(d, g), FP32::class, FloatArray(d * g) { (it % 9 - 4) * 0.05f }),
            initWeightHh = ctx.fromFloatArray(Shape(h, g), FP32::class, FloatArray(h * g) { (it % 7 - 3) * 0.05f }),
            initBiasIh = ctx.fromFloatArray(Shape(g), FP32::class, FloatArray(g) { 0f }),
            initBiasHh = ctx.fromFloatArray(Shape(g), FP32::class, FloatArray(g) { 0f }),
        )
        val input = ctx.fromFloatArray<FP32, Float>(Shape(batch, seq, d), FP32::class, FloatArray(batch * seq * d) { (it % 11 - 5) * 0.1f })
        val out = gru.forward(input, ctx)
        assertEquals(Shape(batch, seq, h), out.shape)
    }
}
