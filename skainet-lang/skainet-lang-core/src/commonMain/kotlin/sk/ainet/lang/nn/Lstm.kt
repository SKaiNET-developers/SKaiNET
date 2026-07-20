package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters

/**
 * Recurrent state of an [Lstm]: hidden state [h] and cell state [c], both `[batch, hiddenSize]`.
 *
 * The state is EXPLICIT and caller-owned so that transducer-style decoders (RNN-T/TDT prediction
 * networks) can advance it one token at a time via [Lstm.step] — and so a single step lowers to a
 * fixed-shape StableHLO graph with the state as plain graph inputs/outputs (StableHLO has no loop
 * construct; see [Gru] for the same unroll-at-trace-time rationale).
 */
public class LstmState<T : DType, V>(
    public val h: Tensor<T, V>,
    public val c: Tensor<T, V>,
) {
    init {
        require(h.rank == 2 && c.rank == 2 && h.shape == c.shape) {
            "LstmState: h and c must both be [batch, hidden], were ${h.shape} / ${c.shape}"
        }
    }
}

/**
 * Single-layer, unidirectional, batch-first LSTM (long short-term memory).
 *
 * Input  `[batch, seq, inputSize]`  ->  output `[batch, seq, hiddenSize]` (all hidden states).
 *
 * Like [Gru], the recurrence is **unrolled over the (static) sequence length at trace time** and
 * built entirely from existing primitive ops (matmul / add / sigmoid / tanh / multiply / narrow /
 * concat), so it runs in the eager engine, is trainable through the standard autodiff tape, and
 * traces to StableHLO with no dedicated converter. For step-wise decoding use [step] with an
 * explicit [LstmState].
 *
 * Gate math matches `torch.nn.LSTM` (gate order **input, forget, cell/g, output**), so PyTorch
 * weights load directly after transposing to the matmul-ready orientation used here:
 *
 *     i = sigmoid(x·W_ii + b_ii + h·W_hi + b_hi)
 *     f = sigmoid(x·W_if + b_if + h·W_hf + b_hf)
 *     g = tanh   (x·W_ig + b_ig + h·W_hg + b_hg)
 *     o = sigmoid(x·W_io + b_io + h·W_ho + b_ho)
 *     c' = f ⊙ c + i ⊙ g
 *     h' = o ⊙ tanh(c')
 *
 * Weights are stored **matmul-ready** (input-major) — the four gates are concatenated on the
 * trailing axis so a single matmul produces all four pre-activations:
 *  - [weightIh] `[inputSize, 4*hiddenSize]`, [weightHh] `[hiddenSize, 4*hiddenSize]`
 *  - [biasIh] / [biasHh] `[4*hiddenSize]`   (gate order i, f, g, o; PyTorch keeps both biases)
 *
 * Stacked LSTMs (e.g. a 2-layer transducer prediction network) compose single-layer cells so the
 * per-layer state stays addressable in [step].
 *
 * @param inputSize  number of input features
 * @param hiddenSize size of the hidden state
 */
public class Lstm<T : DType, V> @kotlin.jvm.JvmOverloads constructor(
    public val inputSize: Int,
    public val hiddenSize: Int,
    override val name: String = "Lstm",
    initWeightIh: Tensor<T, V>,
    initWeightHh: Tensor<T, V>,
    initBiasIh: Tensor<T, V>,
    initBiasHh: Tensor<T, V>,
    public val trainable: Boolean = true,
) : Module<T, V>(), ModuleParameters<T, V> {

    init {
        require(inputSize > 0) { "Lstm($name): inputSize must be positive, was $inputSize" }
        require(hiddenSize > 0) { "Lstm($name): hiddenSize must be positive, was $hiddenSize" }
        val g = 4 * hiddenSize
        fun check2d(t: Tensor<T, V>, rows: Int, cols: Int, what: String) {
            val s = t.shape.dimensions
            require(t.rank == 2 && s[0] == rows && s[1] == cols) {
                "Lstm($name): $what shape must be [$rows, $cols], but was ${t.shape}"
            }
        }
        check2d(initWeightIh, inputSize, g, "weightIh")
        check2d(initWeightHh, hiddenSize, g, "weightHh")
        fun check1d(t: Tensor<T, V>, len: Int, what: String) {
            require(t.rank == 1 && t.shape.dimensions[0] == len) {
                "Lstm($name): $what shape must be [$len], but was ${t.shape}"
            }
        }
        check1d(initBiasIh, g, "biasIh")
        check1d(initBiasHh, g, "biasHh")
    }

    private val pWeightIh = ModuleParameter.WeightParameter("$name.weight_ih", initWeightIh, trainable)
    private val pWeightHh = ModuleParameter.WeightParameter("$name.weight_hh", initWeightHh, trainable)
    private val pBiasIh = ModuleParameter.BiasParameter("$name.bias_ih", initBiasIh, trainable)
    private val pBiasHh = ModuleParameter.BiasParameter("$name.bias_hh", initBiasHh, trainable)

    override val params: List<ModuleParameter<T, V>> = listOf(pWeightIh, pWeightHh, pBiasIh, pBiasHh)

    override val modules: List<Module<T, V>>
        get() = emptyList()

    /** Zero initial state `(h0, c0)` for a batch of [batch] rows. */
    public fun initialState(batch: Int, ctx: ExecutionContext, dtype: kotlin.reflect.KClass<T>): LstmState<T, V> =
        LstmState(
            ctx.zeros(Shape(batch, hiddenSize), dtype),
            ctx.zeros(Shape(batch, hiddenSize), dtype),
        )

    /**
     * One recurrence step: `x_t [batch, inputSize]` + [state] -> `(h' [batch, hiddenSize], state')`.
     * The returned hidden output IS `h'` (also carried inside the new state).
     */
    public fun step(xt: Tensor<T, V>, state: LstmState<T, V>, ctx: ExecutionContext): Pair<Tensor<T, V>, LstmState<T, V>> {
        require(xt.rank == 2 && xt.shape[1] == inputSize) {
            "Lstm($name): step input must be [batch, $inputSize], but was ${xt.shape}"
        }
        val ops = ctx.ops
        val h = hiddenSize

        // gate pre-activations: [batch, 4H], gate order i, f, g, o
        val gx = ops.add(ops.matmul(xt, pWeightIh.value), pBiasIh.value)
        val gh = ops.add(ops.matmul(state.h, pWeightHh.value), pBiasHh.value)
        val gates = ops.add(gx, gh)

        val i = ops.sigmoid(ops.narrow(gates, 1, 0, h))
        val f = ops.sigmoid(ops.narrow(gates, 1, h, h))
        val g = ops.tanh(ops.narrow(gates, 1, 2 * h, h))
        val o = ops.sigmoid(ops.narrow(gates, 1, 3 * h, h))

        val cNew = ops.add(ops.multiply(f, state.c), ops.multiply(i, g))
        val hNew = ops.multiply(o, ops.tanh(cNew))
        return hNew to LstmState(hNew, cNew)
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        require(input.rank == 3) {
            "Lstm($name): input must be 3D [batch, seq, inputSize], but was ${input.shape}"
        }
        val ops = ctx.ops
        val batch = input.shape[0]
        val seq = input.shape[1]

        // Initial state h0 = c0 = 0 (constant leaves in the trace).
        var state = LstmState<T, V>(
            ctx.zeros(Shape(batch, hiddenSize), input.dtype),
            ctx.zeros(Shape(batch, hiddenSize), input.dtype),
        )

        val outputs = ArrayList<Tensor<T, V>>(seq)
        for (t in 0 until seq) {
            // x_t : [batch, inputSize]
            val xt = ops.reshape(ops.narrow(input, 1, t, 1), Shape(batch, inputSize))
            val (h, next) = step(xt, state, ctx)
            state = next
            outputs.add(ops.unsqueeze(h, 1)) // [batch, 1, H]
        }
        return ops.concat(outputs, 1) // [batch, seq, H]
    }
}
