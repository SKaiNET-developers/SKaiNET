package sk.ainet.bench.publish.scenarios

import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.exec.kernel.PanamaVectorQ4KMatmulKernel
import kotlin.random.Random

/**
 * F32 × Q4_K matmul through `PanamaVectorQ4KMatmulKernel`, mirroring
 * `QuantizedMatmulBench`. Primary metric: GOP/s (treating each multiply-add
 * as 2 ops on the dequantized values).
 */
internal class Q4GemmScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-q4-gemm"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gops"
    override val unit: String = "gops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    // smoke: 1024x1024 (smallest LLM-style shape); full: 4096x4096 (hidden->hidden).
    private val inputDim: Int = if (smoke) 1024 else 4096
    private val outputDim: Int = if (smoke) 1024 else 4096
    override val parameters: Map<String, String> = mapOf(
        "input_dim" to inputDim.toString(),
        "output_dim" to outputDim.toString(),
    )

    private lateinit var input: FloatArray
    private lateinit var packedWeights: ByteArray
    private lateinit var output: FloatArray

    override fun setup() {
        require(inputDim % 256 == 0) { "inputDim must be multiple of 256, got $inputDim" }
        val numBlocks = (inputDim / 256) * outputDim
        val rng = Random(42)
        packedWeights = ByteArray(numBlocks * 144)
        rng.nextBytes(packedWeights)
        for (block in 0 until numBlocks) {
            val base = block * 144
            packedWeights[base] = 0x00.toByte(); packedWeights[base + 1] = 0x3C.toByte()
            packedWeights[base + 2] = 0x00.toByte(); packedWeights[base + 3] = 0x3C.toByte()
        }
        input = FloatArray(inputDim) { ((it % 251) - 125).toFloat() / 127f }
        output = FloatArray(outputDim)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        PanamaVectorQ4KMatmulKernel.matmul(
            input, 0,
            packedWeights, 0,
            inputDim, outputDim,
            output, 0,
        )
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = output[0]
        val ops = 2.0 * inputDim.toDouble() * outputDim.toDouble()
        val seconds = elapsedNs / 1_000_000_000.0
        return (ops / seconds) / 1e9
    }
}
