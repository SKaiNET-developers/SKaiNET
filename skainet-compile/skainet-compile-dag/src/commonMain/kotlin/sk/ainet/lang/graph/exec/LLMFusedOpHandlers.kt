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
            // After fusion, edges from absorbed nodes are sorted by destinationInputIndex.
            // gateMatmul and upMatmul share input x, so the layout is:
            //   [x, x, gate_weight, up_weight, down_weight]  (5 inputs)
            // or if x edge is deduplicated:
            //   [x, gate_weight, up_weight, down_weight]  (4 inputs)
            val x: Tensor<DType, Any>
            val gateWeight: Tensor<DType, Any>
            val upWeight: Tensor<DType, Any>
            val downWeight: Tensor<DType, Any>

            if (inputs.size >= 5) {
                x = inputs[0]
                gateWeight = inputs[2]
                upWeight = inputs[3]
                downWeight = inputs[4]
            } else {
                require(inputs.size >= 4) { "fused_swiglu_ffn requires at least 4 inputs, got ${inputs.size}" }
                x = inputs[0]
                gateWeight = inputs[1]
                upWeight = inputs[2]
                downWeight = inputs[3]
            }

            // Weights may need transposing (stored as [out, in] but matmul needs [in, out])
            fun maybeTranspose(input: Tensor<DType, Any>, w: Tensor<DType, Any>): Tensor<DType, Any> {
                val inCols = input.shape[input.rank - 1]
                val wRows = w.shape[0]
                return if (wRows != inCols && w.rank == 2 && w.shape[1] == inCols) {
                    ops.transpose(w)
                } else {
                    w
                }
            }

            // SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
            val gateOut = ops.matmul(x, maybeTranspose(x, gateWeight))
            val gateActivated = ops.silu(gateOut)
            val upOut = ops.matmul(x, maybeTranspose(x, upWeight))
            val gated = ops.multiply(gateActivated, upOut)
            val result = ops.matmul(gated, maybeTranspose(gated, downWeight))

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
            // The fused QKV node receives edges from all 3 original matmul nodes.
            // Each original matmul had 2 inputs (input, weight), and the 3 matmuls
            // shared the same input. After fusion, edges are sorted by destinationInputIndex:
            //   All index-0 edges first: [x, x, x], then index-1 edges: [q_weight, k_weight, v_weight]
            //   Total: [x, x, x, q_weight, k_weight, v_weight]  (6 inputs)
            // Or if edges were deduplicated: [x, q_weight, k_weight, v_weight] (4 inputs)
            val x: Tensor<DType, Any>
            val qWeight: Tensor<DType, Any>
            val kWeight: Tensor<DType, Any>
            val vWeight: Tensor<DType, Any>

            if (inputs.size >= 6) {
                x = inputs[0]
                qWeight = inputs[3]
                kWeight = inputs[4]
                vWeight = inputs[5]
            } else {
                require(inputs.size >= 4) { "fused_qkv_proj requires at least 4 inputs, got ${inputs.size}" }
                x = inputs[0]
                qWeight = inputs[1]
                kWeight = inputs[2]
                vWeight = inputs[3]
            }

            // Weights may need transposing — the original matmul nodes in the graph
            // typically compute x @ W^T. If weight shape doesn't align for matmul,
            // transpose it.
            fun maybeTranspose(w: Tensor<DType, Any>): Tensor<DType, Any> {
                val xCols = x.shape[x.rank - 1]
                val wRows = w.shape[0]
                return if (wRows != xCols && w.rank == 2 && w.shape[1] == xCols) {
                    ops.transpose(w)
                } else {
                    w
                }
            }

            val q = ops.matmul(x, maybeTranspose(qWeight))
            val k = ops.matmul(x, maybeTranspose(kWeight))
            val v = ops.matmul(x, maybeTranspose(vWeight))

            return listOf(q, k, v)
        }
    }
}
