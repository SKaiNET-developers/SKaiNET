package sk.ainet.lang.nn.layers

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP64
import kotlin.random.Random

/**
 * Dropout layer with inverted-dropout semantics, aware of ExecutionContext phases.
 *
 * During training ([ExecutionContext.inTraining] and [training] both true, `p > 0`),
 * each element is zeroed with probability [p] and the surviving elements are
 * scaled by `1 / (1 - p)` so the expected activation stays constant. In the
 * evaluation phase (or with [training] disabled) the layer is the identity —
 * no rescaling is needed at inference time.
 *
 * The mask is a constant tensor combined with an element-wise multiply, so
 * gradients flow to the surviving inputs only — the standard dropout backward —
 * without needing a dedicated autograd rule.
 *
 * @param p probability of zeroing an element, in `[0, 1)`
 * @param training manual override; set to false to disable masking regardless of phase
 * @param name name of the module
 * @param random RNG for the mask — inject a seeded [Random] for reproducible runs
 */
public class Dropout<T : DType, V>(
    public val p: Float = 0.5f,
    public var training: Boolean = true,
    override val name: String = "Dropout",
    private val random: Random = Random.Default,
) : Module<T, V>() {

    init {
        require(p >= 0f) { "Dropout($name): p must be >= 0, was $p" }
        require(p < 1f) { "Dropout($name): p must be < 1 to avoid division by zero, was $p" }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    /**
     * Context-aware forward: stochastic masking when ctx is in the training
     * phase, identity otherwise. Hooks are dispatched if available.
     */
    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            if (!ctx.inTraining || !training || p == 0f) {
                input
            } else {
                val mask = ctx.fromData(bernoulliMask(input, ctx), input.dtype)
                ctx.ops.multiply(input, mask)
            }
        }

    private fun bernoulliMask(input: Tensor<T, V>, ctx: ExecutionContext): TensorData<T, V> {
        val scale = 1f / (1f - p)
        return ctx.tensorDataFactory.init(input.shape, input.dtype) {
            val keep = random.nextFloat() >= p
            @Suppress("UNCHECKED_CAST")
            when (input.dtype) {
                FP32::class, FP16::class, BF16::class -> (if (keep) scale else 0f) as V
                FP64::class -> (if (keep) scale.toDouble() else 0.0) as V
                else -> throw UnsupportedOperationException(
                    "Dropout($name): unsupported dtype ${input.dtype} — floating-point tensors only"
                )
            }
        }
    }
}
