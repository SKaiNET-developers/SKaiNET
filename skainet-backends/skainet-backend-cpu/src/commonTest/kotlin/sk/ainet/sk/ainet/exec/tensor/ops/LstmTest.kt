package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.math.exp
import kotlin.math.tanh
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.Lstm
import sk.ainet.lang.nn.LstmState
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/**
 * Eager forward correctness for the LSTM layer, checked against an independent scalar
 * reference (raw FloatArray loops) computing the same PyTorch i,f,g,o recurrence.
 * Exercises all four gates, the cell-state feedback, and the step()/sequence equivalence
 * (the transducer prediction-network usage pattern).
 */
class LstmTest {
    private val ctx = DirectCpuExecutionContext()

    private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

    // Independent reference: input [S*D] (batch=1), weights matmul-ready (row-major).
    private fun lstmRef(
        x: FloatArray, seq: Int, d: Int, h: Int,
        wIh: FloatArray, wHh: FloatArray, bIh: FloatArray, bHh: FloatArray,
    ): FloatArray {
        val g4 = 4 * h
        val hidden = FloatArray(h)
        val cell = FloatArray(h)
        val out = FloatArray(seq * h)
        for (t in 0 until seq) {
            val gx = FloatArray(g4) { k -> bIh[k] + (0 until d).sumOf { i -> (x[t * d + i] * wIh[i * g4 + k]).toDouble() }.toFloat() }
            val gh = FloatArray(g4) { k -> bHh[k] + (0 until h).sumOf { j -> (hidden[j] * wHh[j * g4 + k]).toDouble() }.toFloat() }
            for (j in 0 until h) {
                val i = sigmoid(gx[j] + gh[j])                                   // input gate
                val f = sigmoid(gx[h + j] + gh[h + j])                           // forget gate
                val g = tanh((gx[2 * h + j] + gh[2 * h + j]).toDouble()).toFloat() // cell candidate
                val o = sigmoid(gx[3 * h + j] + gh[3 * h + j])                   // output gate
                cell[j] = f * cell[j] + i * g
                hidden[j] = o * tanh(cell[j].toDouble()).toFloat()
            }
            for (j in 0 until h) out[t * h + j] = hidden[j]
        }
        return out
    }

    private fun makeLstm(d: Int, h: Int, wIh: FloatArray, wHh: FloatArray, bIh: FloatArray, bHh: FloatArray) =
        Lstm<FP32, Float>(
            inputSize = d, hiddenSize = h, name = "lstm",
            initWeightIh = ctx.fromFloatArray(Shape(d, 4 * h), FP32::class, wIh),
            initWeightHh = ctx.fromFloatArray(Shape(h, 4 * h), FP32::class, wHh),
            initBiasIh = ctx.fromFloatArray(Shape(4 * h), FP32::class, bIh),
            initBiasHh = ctx.fromFloatArray(Shape(4 * h), FP32::class, bHh),
        )

    @Test
    fun lstm_forward_matches_reference() {
        val batch = 1; val seq = 3; val d = 2; val h = 2; val g4 = 4 * h
        // deterministic small weights/inputs (kept in sigmoid/tanh's sensitive range)
        val x = FloatArray(seq * d) { ((it % 5) - 2) * 0.3f }
        val wIh = FloatArray(d * g4) { ((it % 7) - 3) * 0.1f }
        val wHh = FloatArray(h * g4) { ((it % 5) - 2) * 0.15f }
        val bIh = FloatArray(g4) { ((it % 3) - 1) * 0.2f }
        val bHh = FloatArray(g4) { ((it % 4) - 2) * 0.1f }

        val lstm = makeLstm(d, h, wIh, wHh, bIh, bHh)
        val input = ctx.fromFloatArray<FP32, Float>(Shape(batch, seq, d), FP32::class, x)
        val out = lstm.forward(input, ctx)
        assertEquals(Shape(batch, seq, h), out.shape)

        val expected = lstmRef(x, seq, d, h, wIh, wHh, bIh, bHh)
        for (t in 0 until seq) for (j in 0 until h) {
            assertEquals(expected[t * h + j], out.data[0, t, j], 1e-5f)
        }
    }

    @Test
    fun lstm_step_equals_unrolled_forward() {
        // The transducer usage: repeated step() with caller-owned state must reproduce the
        // sequence forward exactly.
        val batch = 1; val seq = 4; val d = 3; val h = 2; val g4 = 4 * h
        val x = FloatArray(seq * d) { ((it % 6) - 3) * 0.25f }
        val wIh = FloatArray(d * g4) { ((it % 9) - 4) * 0.08f }
        val wHh = FloatArray(h * g4) { ((it % 7) - 3) * 0.12f }
        val bIh = FloatArray(g4) { ((it % 5) - 2) * 0.1f }
        val bHh = FloatArray(g4) { ((it % 3) - 1) * 0.15f }

        val lstm = makeLstm(d, h, wIh, wHh, bIh, bHh)
        val seqOut = lstm.forward(ctx.fromFloatArray<FP32, Float>(Shape(batch, seq, d), FP32::class, x), ctx)

        var state = lstm.initialState(batch, ctx, FP32::class)
        for (t in 0 until seq) {
            val xt = ctx.fromFloatArray<FP32, Float>(Shape(batch, d), FP32::class, FloatArray(d) { x[t * d + it] })
            val (out, next) = lstm.step(xt, state, ctx)
            state = next
            for (j in 0 until h) {
                assertEquals(seqOut.data[0, t, j], out.data[0, j], 1e-6f)
            }
        }
    }

    @Test
    fun lstm_output_shape_is_batch_seq_hidden() {
        val batch = 2; val seq = 3; val d = 4; val h = 5; val g4 = 4 * h
        val lstm = makeLstm(
            d, h,
            FloatArray(d * g4) { (it % 9 - 4) * 0.05f },
            FloatArray(h * g4) { (it % 7 - 3) * 0.05f },
            FloatArray(g4) { 0f },
            FloatArray(g4) { 0f },
        )
        val input = ctx.fromFloatArray<FP32, Float>(Shape(batch, seq, d), FP32::class, FloatArray(batch * seq * d) { (it % 11 - 5) * 0.1f })
        val out = lstm.forward(input, ctx)
        assertEquals(Shape(batch, seq, h), out.shape)
    }

    @Test
    fun forget_gate_controls_cell_memory() {
        // With strongly positive forget-gate bias and zero input gate, the cell must persist;
        // sanity-check the gate ORDER (i,f,g,o) is honored: a big b_ih in the f-slot must not
        // leak into i/g/o behavior.
        val d = 1; val h = 1; val g4 = 4
        val bIh = floatArrayOf(-20f, 20f, 0f, 0f)   // i≈0, f≈1, g=0, o≈0.5
        val lstm = makeLstm(d, h, FloatArray(d * g4), FloatArray(h * g4), bIh, FloatArray(g4))
        val state0 = LstmState(
            ctx.fromFloatArray<FP32, Float>(Shape(1, 1), FP32::class, floatArrayOf(0f)),
            ctx.fromFloatArray<FP32, Float>(Shape(1, 1), FP32::class, floatArrayOf(0.8f)),  // pre-charged cell
        )
        val (out, next) = lstm.step(ctx.fromFloatArray<FP32, Float>(Shape(1, 1), FP32::class, floatArrayOf(1f)), state0, ctx)
        // c' ≈ 1.0*0.8 + 0*0 = 0.8 ; h' ≈ 0.5 * tanh(0.8)
        assertEquals(0.8f, next.c.data[0, 0], 1e-4f)
        assertEquals(0.5f * tanh(0.8), out.data[0, 0].toDouble().toFloat().toDouble(), 1e-4)
    }
}
