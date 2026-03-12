package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Multi-Head Attention module.
 *
 * Supports:
 * - Grouped-Query Attention (nKVHeads < nHeads)
 * - Optional QK-Norm (per-head RMSNorm on Q and K before attention)
 * - Optional RoPE (rotary position embeddings)
 * - Optional KV Cache (autoregressive decoding)
 * - Causal or bidirectional masking
 *
 * Weight parameters are exposed as ModuleParameters and loaded via WeightMapper.
 * This module does NOT use Linear submodules because LLM projections often omit bias.
 *
 * @param dim model dimension
 * @param nHeads number of query heads
 * @param nKVHeads number of key-value heads (for GQA; defaults to nHeads)
 * @param causal whether to apply causal masking
 * @param qkNorm whether to apply per-head RMSNorm on Q and K
 * @param bias whether Q/K/V/O projections have bias (true for BERT, false for Llama)
 * @param name module name
 */
public class MultiHeadAttention<T : DType, V>(
    public val dim: Int,
    public val nHeads: Int,
    public val nKVHeads: Int = nHeads,
    public val causal: Boolean = true,
    public val qkNorm: Boolean = false,
    public val bias: Boolean = false,
    override val name: String = "MultiHeadAttention",
    public var rope: RoPE<T, V>? = null,
    public var kvCache: KVCache<T, V>? = null
) : Module<T, V>(), ModuleParameters<T, V> {

    public val headDim: Int = dim / nHeads
    public val kvDim: Int = headDim * nKVHeads

    // Weight parameters — placeholder tensors, replaced by WeightMapper
    @Suppress("UNCHECKED_CAST")
    private fun voidWeight(paramName: String, rows: Int, cols: Int): ModuleParameter<T, V> {
        val tensor = VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(rows, cols)
                override fun get(vararg indices: Int): V = 0.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            Any::class as KClass<T>
        )
        return ModuleParameter.WeightParameter(paramName, tensor)
    }

    @Suppress("UNCHECKED_CAST")
    private fun voidBias(paramName: String, size: Int): ModuleParameter<T, V> {
        val tensor = VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(size)
                override fun get(vararg indices: Int): V = 0.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            Any::class as KClass<T>
        )
        return ModuleParameter.BiasParameter(paramName, tensor)
    }

    // Indices into params list depend on whether bias is enabled.
    // With bias=false: [qW, kW, vW, oW]
    // With bias=true:  [qW, qB, kW, kB, vW, vB, oW, oB]
    private val qWIdx = 0
    private val kWIdx = if (bias) 2 else 1
    private val vWIdx = if (bias) 4 else 2
    private val oWIdx = if (bias) 6 else 3

    override val params: List<ModuleParameter<T, V>> = buildList {
        add(voidWeight("$name.q_proj.weight", dim, dim))
        if (bias) add(voidBias("$name.q_proj.bias", dim))
        add(voidWeight("$name.k_proj.weight", kvDim, dim))
        if (bias) add(voidBias("$name.k_proj.bias", kvDim))
        add(voidWeight("$name.v_proj.weight", kvDim, dim))
        if (bias) add(voidBias("$name.v_proj.bias", kvDim))
        add(voidWeight("$name.o_proj.weight", dim, dim))
        if (bias) add(voidBias("$name.o_proj.bias", dim))
    }

    // Optional QK-Norm layers
    public val qNorm: RMSNormalization<T, V>? = if (qkNorm) {
        RMSNormalization(intArrayOf(headDim), name = "$name.q_norm")
    } else null

    public val kNorm: RMSNormalization<T, V>? = if (qkNorm) {
        RMSNormalization(intArrayOf(headDim), name = "$name.k_norm")
    } else null

    @Suppress("UNCHECKED_CAST")
    override val modules: List<Module<T, V>>
        get() = buildList {
            if (qNorm != null) add(qNorm)
            if (kNorm != null) add(kNorm)
            if (rope != null) add(rope as Module<T, V>)
            if (kvCache != null) add(kvCache as Module<T, V>)
        }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val scale = 1.0f / sqrt(headDim.toFloat())

        val wQ = params[qWIdx].value
        val wK = params[kWIdx].value
        val wV = params[vWIdx].value
        val wO = params[oWIdx].value

        // Project Q, K, V: input @ W^T (+ bias if enabled)
        var q = ops.matmul(input, ops.transpose(wQ))
        var k = ops.matmul(input, ops.transpose(wK))
        var v = ops.matmul(input, ops.transpose(wV))

        if (bias) {
            q = ops.add(q, params[qWIdx + 1].value)
            k = ops.add(k, params[kWIdx + 1].value)
            v = ops.add(v, params[vWIdx + 1].value)
        }

        // Reshape to multi-head: [seqLen, dim] → [nHeads, seqLen, headDim]
        val seqLen = if (input.rank >= 2) input.shape[input.rank - 2] else 1
        q = ops.reshape(q, Shape(nHeads, seqLen, headDim))
        k = ops.reshape(k, Shape(nKVHeads, seqLen, headDim))
        val vReshaped = ops.reshape(v, Shape(nKVHeads, seqLen, headDim))

        // Optional QK-Norm
        if (qNorm != null && kNorm != null) {
            q = qNorm.forward(q, ctx)
            k = kNorm.forward(k, ctx)
        }

        // Optional RoPE
        val ropeModule = rope
        if (ropeModule != null) {
            val position = kvCache?.position ?: 0
            q = ropeModule.forward(q, position, ctx)
            k = ropeModule.forward(k, position, ctx)
        }

        // Optional KV Cache update
        val (fullK, fullV) = if (kvCache != null) {
            kvCache!!.update(k, vReshaped, ctx)
        } else {
            k to vReshaped
        }

        // Expand KV heads for GQA if needed
        val expandedK = if (nKVHeads < nHeads) repeatKVHeads(fullK, nHeads / nKVHeads, ops) else fullK
        val expandedV = if (nKVHeads < nHeads) repeatKVHeads(fullV, nHeads / nKVHeads, ops) else fullV

        // Unsqueeze batch dim for SDPA: [1, nHeads, seqLen, headDim]
        val qBatched = ops.unsqueeze(q, 0)
        val kBatched = ops.unsqueeze(expandedK, 0)
        val vBatched = ops.unsqueeze(expandedV, 0)

        // Scaled dot-product attention
        val attnOut = ops.scaledDotProductAttention(
            query = qBatched,
            key = kBatched,
            value = vBatched,
            mask = null,
            scale = scale,
            causal = causal
        )

        // Remove batch dim and merge heads: [1, nHeads, seqLen, headDim] → [seqLen, dim]
        val squeezed = ops.squeeze(attnOut, 0)
        val merged = ops.reshape(squeezed, Shape(seqLen, dim))

        // Output projection: merged @ wO^T (+ bias if enabled)
        var output = ops.matmul(merged, ops.transpose(wO))
        if (bias) {
            output = ops.add(output, params[oWIdx + 1].value)
        }
        return output
    }

    private fun repeatKVHeads(t: Tensor<T, V>, repeats: Int, ops: sk.ainet.lang.tensor.ops.TensorOps): Tensor<T, V> {
        if (repeats == 1) return t
        val repeated = (0 until repeats).map { t }
        return ops.concat(repeated, dim = 0)
    }
}
