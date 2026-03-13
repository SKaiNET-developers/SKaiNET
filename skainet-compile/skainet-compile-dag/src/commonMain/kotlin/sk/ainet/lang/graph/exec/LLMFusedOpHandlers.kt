package sk.ainet.lang.graph.exec

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType

/**
 * CPU fallback implementations for fused LLM operations.
 *
 * These handlers decompose fused ops back into sequences of [TensorOps] calls.
 * They produce correct results on any backend but don't provide the performance
 * benefit of a true fused kernel. Platform-specific backends (Metal, CUDA) should
 * register their own handlers to override these.
 *
 * Register via:
 * ```kotlin
 * LLMFusedOpHandlers.registerAll()
 * ```
 */
public object LLMFusedOpHandlers {

    /**
     * Register all CPU fallback handlers with [ComputeGraphExecutor].
     * Call once at application startup.
     */
    public fun registerAll() {
        ComputeGraphExecutor.registerFusedOp("fused_rms_norm", RmsNormHandler)
        ComputeGraphExecutor.registerFusedOp("fused_swiglu_ffn", SwiGluFFNHandler)
        ComputeGraphExecutor.registerFusedOp("fused_qkv_proj", QKVProjHandler)
    }

    /**
     * Fused RMSNorm: x * weight / sqrt(mean(x^2) + eps)
     *
     * Decomposed from: multiply(x,x) → mean → add(eps) → sqrt → rdiv → multiply → multiply(weight)
     *
     * Inputs:
     *  - [0] x: the input tensor
     *  - [1] weight: the learned scale parameter
     *
     * Params:
     *  - "eps": epsilon for numerical stability (default 1e-5)
     */
    private object RmsNormHandler : FusedOpHandler<DType, Any> {
        override fun execute(
            ops: TensorOps,
            inputs: List<Tensor<DType, Any>>,
            params: Map<String, Any>
        ): List<Tensor<DType, Any>> {
            require(inputs.size >= 1) { "fused_rms_norm requires at least 1 input (x), got ${inputs.size}" }
            val x = inputs[0]
            val eps = (params["eps"] as? Number)?.toFloat() ?: 1e-5f

            // RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
            val xSquared = ops.multiply(x, x)
            val meanSquared = ops.mean(xSquared, -1)  // mean over last dim
            val meanPlusEps = ops.addScalar(meanSquared, eps)
            val rms = ops.sqrt(meanPlusEps)
            val normalized = ops.divide(x, rms)

            return if (inputs.size >= 2) {
                val weight = inputs[1]
                listOf(ops.multiply(normalized, weight))
            } else {
                listOf(normalized)
            }
        }
    }

    /**
     * Fused SwiGLU FFN: down(silu(gate(x)) * up(x))
     *
     * Decomposed from: matmul(gate) → silu → multiply(matmul(up)) → matmul(down)
     *
     * Inputs:
     *  - [0] x: the input tensor (from norm output)
     *  - [1] gate_weight: gate projection weight matrix
     *  - [2] up_weight: up projection weight matrix
     *  - [3] down_weight: down projection weight matrix
     */
    private object SwiGluFFNHandler : FusedOpHandler<DType, Any> {
        override fun execute(
            ops: TensorOps,
            inputs: List<Tensor<DType, Any>>,
            params: Map<String, Any>
        ): List<Tensor<DType, Any>> {
            require(inputs.size >= 4) { "fused_swiglu_ffn requires 4 inputs (x, gate, up, down), got ${inputs.size}" }
            val x = inputs[0]
            val gateWeight = inputs[1]
            val upWeight = inputs[2]
            val downWeight = inputs[3]

            // SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
            val gateOut = ops.matmul(x, gateWeight)
            val gateActivated = ops.silu(gateOut)
            val upOut = ops.matmul(x, upWeight)
            val gated = ops.multiply(gateActivated, upOut)
            val result = ops.matmul(gated, downWeight)

            return listOf(result)
        }
    }

    /**
     * Fused QKV Projection: single batched matmul producing [Q, K, V].
     *
     * Decomposed from: 3 separate matmul nodes sharing the same input.
     *
     * Inputs:
     *  - [0] x: the input tensor (from norm output)
     *  - [1] q_weight: query projection weight
     *  - [2] k_weight: key projection weight
     *  - [3] v_weight: value projection weight
     *
     * Returns 3 tensors: [Q, K, V]
     *
     * A true fused implementation would concatenate the weight matrices and
     * do a single matmul + split, reducing memory bandwidth.
     */
    private object QKVProjHandler : FusedOpHandler<DType, Any> {
        override fun execute(
            ops: TensorOps,
            inputs: List<Tensor<DType, Any>>,
            params: Map<String, Any>
        ): List<Tensor<DType, Any>> {
            require(inputs.size >= 4) { "fused_qkv_proj requires 4 inputs (x, q, k, v weights), got ${inputs.size}" }
            val x = inputs[0]
            val qWeight = inputs[1]
            val kWeight = inputs[2]
            val vWeight = inputs[3]

            // Decomposed: 3 independent matmuls
            val q = ops.matmul(x, qWeight)
            val k = ops.matmul(x, kWeight)
            val v = ops.matmul(x, vWeight)

            return listOf(q, k, v)
        }
    }
}
