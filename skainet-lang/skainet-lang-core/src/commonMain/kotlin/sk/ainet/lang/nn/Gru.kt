package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters

/**
 * Single-layer, unidirectional, batch-first GRU (gated recurrent unit).
 *
 * Input  `[batch, seq, inputSize]`  ->  output `[batch, seq, hiddenSize]` (all hidden states).
 *
 * The recurrence is **unrolled over the (static) sequence length at trace time** and built
 * entirely from existing primitive ops (matmul / add / sigmoid / tanh / multiply / narrow /
 * concat), so it runs in the eager engine, is trainable through the standard autodiff tape,
 * and traces to StableHLO with no dedicated converter (StableHLO has no loop construct).
 *
 * Gate math matches `torch.nn.GRU` (gate order reset, update, new), so PyTorch weights load
 * directly (after transposing to the matmul-ready orientation used here):
 *
 *     r = sigmoid(x·W_ir + b_ir + h·W_hr + b_hr)
 *     z = sigmoid(x·W_iz + b_iz + h·W_hz + b_hz)
 *     n = tanh   (x·W_in + b_in + r ⊙ (h·W_hn + b_hn))
 *     h' = (1 - z) ⊙ n + z ⊙ h
 *
 * Weights are stored **matmul-ready** (input-major) — the three gates are concatenated on the
 * trailing axis so a single matmul produces all three pre-activations:
 *  - [weightIh] `[inputSize, 3*hiddenSize]`, [weightHh] `[hiddenSize, 3*hiddenSize]`
 *  - [biasIh] / [biasHh] `[3*hiddenSize]`   (gate order r, z, n)
 *
 * @param inputSize  number of input features
 * @param hiddenSize size of the hidden state
 */
public class Gru<T : DType, V> @kotlin.jvm.JvmOverloads constructor(
    public val inputSize: Int,
    public val hiddenSize: Int,
    override val name: String = "Gru",
    initWeightIh: Tensor<T, V>,
    initWeightHh: Tensor<T, V>,
    initBiasIh: Tensor<T, V>,
    initBiasHh: Tensor<T, V>,
    public val trainable: Boolean = true,
) : Module<T, V>(), ModuleParameters<T, V> {

    init {
        require(inputSize > 0) { "Gru($name): inputSize must be positive, was $inputSize" }
        require(hiddenSize > 0) { "Gru($name): hiddenSize must be positive, was $hiddenSize" }
        val g = 3 * hiddenSize
        fun check2d(t: Tensor<T, V>, rows: Int, cols: Int, what: String) {
            val s = t.shape.dimensions
            require(t.rank == 2 && s[0] == rows && s[1] == cols) {
                "Gru($name): $what shape must be [$rows, $cols], but was ${t.shape}"
            }
        }
        check2d(initWeightIh, inputSize, g, "weightIh")
        check2d(initWeightHh, hiddenSize, g, "weightHh")
        fun check1d(t: Tensor<T, V>, len: Int, what: String) {
            require(t.rank == 1 && t.shape.dimensions[0] == len) {
                "Gru($name): $what shape must be [$len], but was ${t.shape}"
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

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        require(input.rank == 3) {
            "Gru($name): input must be 3D [batch, seq, inputSize], but was ${input.shape}"
        }
        val ops = ctx.ops
        val batch = input.shape[0]
        val seq = input.shape[1]
        val h = hiddenSize

        val weightIh = pWeightIh.value
        val weightHh = pWeightHh.value
        val biasIh = pBiasIh.value
        val biasHh = pBiasHh.value

        // Initial hidden state h0 = 0 (a constant leaf in the trace).
        var hidden = ctx.zeros<T, V>(Shape(batch, h), input.dtype)

        val outputs = ArrayList<Tensor<T, V>>(seq)
        for (t in 0 until seq) {
            // x_t : [batch, inputSize]
            val xt = ops.reshape(ops.narrow(input, 1, t, 1), Shape(batch, inputSize))
            // gate pre-activations: [batch, 3H]
            val gx = ops.add(ops.matmul(xt, weightIh), biasIh)
            val gh = ops.add(ops.matmul(hidden, weightHh), biasHh)
            // split into reset / update / new on the gate axis
            val xr = ops.narrow(gx, 1, 0, h); val hr = ops.narrow(gh, 1, 0, h)
            val xz = ops.narrow(gx, 1, h, h); val hz = ops.narrow(gh, 1, h, h)
            val xn = ops.narrow(gx, 1, 2 * h, h); val hn = ops.narrow(gh, 1, 2 * h, h)

            val r = ops.sigmoid(ops.add(xr, hr))
            val z = ops.sigmoid(ops.add(xz, hz))
            val n = ops.tanh(ops.add(xn, ops.multiply(r, hn)))
            // h' = (1 - z) ⊙ n + z ⊙ h
            val oneMinusZ = ops.rsubScalar(1.0, z)
            hidden = ops.add(ops.multiply(oneMinusZ, n), ops.multiply(z, hidden))

            outputs.add(ops.unsqueeze(hidden, 1)) // [batch, 1, H]
        }
        return ops.concat(outputs, 1) // [batch, seq, H]
    }
}
