package sk.ainet.models.llama

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Optional graph-level acceleration for LLaMA transformer layers.
 *
 * When provided to [LlamaRuntime], compiled graph implementations
 * replace individual matmul calls with fused graph executions,
 * eliminating per-op synchronization overhead on GPU backends.
 *
 * Implementations are platform-specific (e.g. MPSGraph on Apple).
 */
public interface GraphAccelerator<T : DType> {

    public data class QKVResult<T : DType>(
        val q: Tensor<T, Float>,
        val k: Tensor<T, Float>,
        val v: Tensor<T, Float>
    )

    /**
     * Run the fused RMSNorm + QKV projection for a layer.
     *
     * Replaces: attnNorm(input) then 3 separate matmuls.
     * Returns null if the graph is not compiled for this layer
     * or execution fails (caller should fall back to individual ops).
     */
    public fun runQKV(layerIdx: Int, input: Tensor<T, Float>): QKVResult<T>?

    /**
     * Run the fused RMSNorm + FFN block + residual for a layer.
     *
     * Replaces: ffnNorm(input) then gate/up/down matmuls + silu + residual add.
     * Returns null if the graph is not compiled for this layer
     * or execution fails (caller should fall back to individual ops).
     */
    public fun runFFN(layerIdx: Int, input: Tensor<T, Float>): Tensor<T, Float>?

    public fun close()
}
