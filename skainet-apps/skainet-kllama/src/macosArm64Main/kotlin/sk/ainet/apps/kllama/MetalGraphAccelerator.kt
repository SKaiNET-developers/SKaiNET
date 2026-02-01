package sk.ainet.apps.kllama

import sk.ainet.exec.tensor.ops.CompiledFFNProjection
import sk.ainet.exec.tensor.ops.CompiledQKVProjection
import sk.ainet.exec.tensor.ops.MetalTensorOps
import sk.ainet.io.gguf.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * MPSGraph-backed [GraphAccelerator] for Apple platforms.
 *
 * Builds compiled QKV and FFN graphs for each transformer layer,
 * replacing 6 separate matmul synchronization points per layer with
 * 2 fused graph executions. For a 24-layer model this eliminates
 * ~144 command buffer flushes per token.
 */
internal class MetalGraphAccelerator<T : DType>(
    private val qkvGraphs: List<CompiledQKVProjection<T>?>,
    private val ffnGraphs: List<CompiledFFNProjection<T>?>
) : GraphAccelerator<T> {

    override fun runQKV(layerIdx: Int, input: Tensor<T, Float>): GraphAccelerator.QKVResult<T>? {
        val graph = qkvGraphs.getOrNull(layerIdx) ?: return null
        val result = graph.execute(input) ?: return null
        if (layerIdx == 0 && debugCount < 2) {
            debugCount++
            val inputBuf = input.data.copyToFloatArray()
            val qBuf = result.q.data.copyToFloatArray()
            println("[DEBUG] QKV layer0: input[0..3]=${inputBuf.take(4)} " +
                "inputNorm=${inputBuf.map { it * it }.sum()} " +
                "q[0..3]=${qBuf.take(4)} qNorm=${qBuf.map { it * it }.sum()}")
        }
        return GraphAccelerator.QKVResult(result.q, result.k, result.v)
    }

    private var debugCount = 0

    override fun runFFN(layerIdx: Int, input: Tensor<T, Float>): Tensor<T, Float>? {
        val graph = ffnGraphs.getOrNull(layerIdx) ?: return null
        return graph.execute(input)
    }

    override fun close() {
        qkvGraphs.forEach { it?.close() }
        ffnGraphs.forEach { it?.close() }
    }

    companion object {
        /**
         * Build compiled graphs for all transformer layers.
         *
         * @return MetalGraphAccelerator or null if graph compilation fails
         */
        fun <T : DType> build(
            weights: LlamaRuntimeWeights<T>,
            ops: MetalTensorOps,
            dtype: KClass<T>,
            eps: Float = 1e-5f
        ): MetalGraphAccelerator<T>? {
            val dim = weights.metadata.embeddingLength
            val kvHeadCount = weights.metadata.kvHeadCount
            val headSize = dim / weights.metadata.headCount
            val kvDim = kvHeadCount * headSize

            val inputShape = Shape(1, dim)
            val qShape = Shape(1, dim)
            val kShape = Shape(1, kvDim)
            val vShape = Shape(1, kvDim)

            val qkvGraphs = mutableListOf<CompiledQKVProjection<T>?>()
            val ffnGraphs = mutableListOf<CompiledFFNProjection<T>?>()

            for ((i, layer) in weights.layers.withIndex()) {
                if (i == 0) {
                    val wqBuf = layer.wq.data.copyToFloatArray()
                    val normBuf = layer.attnNorm.data.copyToFloatArray()
                    println("[DEBUG] Layer0 wq[0..3]=${wqBuf.take(4)} wqNorm=${wqBuf.take(100).map{it*it}.sum()} " +
                        "wqShape=${layer.wq.shape} normW[0..3]=${normBuf.take(4)} normShape=${layer.attnNorm.shape}")
                }
                val qkv = CompiledQKVProjection.build(
                    wq = layer.wq,
                    wk = layer.wk,
                    wv = layer.wv,
                    attnNormWeight = layer.attnNorm,
                    eps = eps,
                    inputShape = inputShape,
                    qShape = qShape,
                    kShape = kShape,
                    vShape = vShape,
                    dtype = dtype,
                    ops = ops
                )
                if (qkv == null) {
                    println("Warning: QKV graph compilation failed for layer $i, falling back to individual ops")
                }
                qkvGraphs.add(qkv)

                val ffn = CompiledFFNProjection.build(
                    ffnGate = layer.ffnGate,
                    ffnUp = layer.ffnUp,
                    ffnDown = layer.ffnDown,
                    ffnNormWeight = layer.ffnNorm,
                    eps = eps,
                    inputShape = inputShape,
                    dtype = dtype,
                    ops = ops
                )
                if (ffn == null) {
                    println("Warning: FFN graph compilation failed for layer $i, falling back to individual ops")
                }
                ffnGraphs.add(ffn)
            }

            val compiledQKV = qkvGraphs.count { it != null }
            val compiledFFN = ffnGraphs.count { it != null }
            val total = weights.layers.size

            if (compiledQKV == 0 && compiledFFN == 0) {
                println("Warning: No graphs compiled successfully, graph acceleration disabled")
                return null
            }

            println("MPSGraph: compiled $compiledQKV/$total QKV + $compiledFFN/$total FFN graphs")
            return MetalGraphAccelerator(qkvGraphs, ffnGraphs)
        }
    }
}
