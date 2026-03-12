package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * Rotary Position Embedding (RoPE) module.
 *
 * Precomputes cos/sin frequency tables and applies rotary embeddings to input tensors.
 * Used by Llama, Apertus, and other decoder architectures.
 *
 * Input shape: [seqLen, dim] or [batch, seqLen, dim]
 * The last dimension is split into pairs for rotation.
 *
 * @param headDim dimension of each attention head (must be even)
 * @param maxSeqLen maximum sequence length for precomputed tables
 * @param base RoPE base frequency (default 10000.0)
 * @param name module name
 */
public class RoPE<T : DType, V>(
    public val headDim: Int,
    public val maxSeqLen: Int,
    private val base: Float = 10000.0f,
    override val name: String = "RoPE"
) : Module<T, V>() {

    init {
        require(headDim % 2 == 0) { "RoPE headDim must be even, got $headDim" }
    }

    override val modules: List<Module<T, V>> = emptyList()

    // Precomputed frequency tables: [maxSeqLen, headDim/2]
    private val cosTable: FloatArray = FloatArray(maxSeqLen * (headDim / 2))
    private val sinTable: FloatArray = FloatArray(maxSeqLen * (headDim / 2))

    init {
        val halfDim = headDim / 2
        for (pos in 0 until maxSeqLen) {
            for (i in 0 until halfDim) {
                val freq = 1.0f / base.pow(2.0f * i / headDim)
                val angle = pos * freq
                cosTable[pos * halfDim + i] = cos(angle)
                sinTable[pos * halfDim + i] = sin(angle)
            }
        }
    }

    /**
     * Apply rotary embeddings to [input] starting at [position].
     *
     * @param input tensor to rotate, last dim = headDim
     * @param position the starting position index (for autoregressive decoding)
     * @param ctx execution context
     * @return rotated tensor with same shape
     */
    public fun forward(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        return sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            applyRoPE(input, position, ctx)
        }
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        // Default forward with position=0 (for tracing / non-autoregressive use)
        return applyRoPE(input, 0, ctx)
    }

    private fun applyRoPE(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val halfDim = headDim / 2
        // Split last dimension into even/odd halves
        val splits = ops.split(input, halfDim, dim = input.rank - 1)
        val even = splits[0]
        val odd = splits[1]

        // Build cos/sin tensors for current position range
        val seqLen = input.shape[input.rank - 2]
        val cosData = FloatArray(seqLen * halfDim)
        val sinData = FloatArray(seqLen * halfDim)
        for (s in 0 until seqLen) {
            val pos = position + s
            for (i in 0 until halfDim) {
                cosData[s * halfDim + i] = cosTable[pos * halfDim + i]
                sinData[s * halfDim + i] = sinTable[pos * halfDim + i]
            }
        }

        val cosShape = Shape(seqLen, halfDim)
        @Suppress("UNCHECKED_CAST")
        val cosTensor = ctx.fromFloatArray(cosShape, input.dtype, cosData) as Tensor<T, V>
        @Suppress("UNCHECKED_CAST")
        val sinTensor = ctx.fromFloatArray(cosShape, input.dtype, sinData) as Tensor<T, V>

        // Rotation: rotEven = even * cos - odd * sin
        //           rotOdd  = odd * cos + even * sin
        val rotEven = ops.subtract(ops.multiply(even, cosTensor), ops.multiply(odd, sinTensor))
        val rotOdd = ops.add(ops.multiply(odd, cosTensor), ops.multiply(even, sinTensor))

        // Concatenate back
        return ops.concat(listOf(rotEven, rotOdd), dim = input.rank - 1)
    }
}
