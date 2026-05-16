package sk.ainet.bench.publish.scenarios

import sk.ainet.backend.api.kernel.Q8_0MatmulKernel
import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.exec.kernel.PanamaVectorQ8_0MatmulKernel
import sk.ainet.exec.kernel.ScalarQ8_0MatmulKernel
import kotlin.random.Random

/**
 * F32 input × Q8_0-packed weight matrix-vector multiply through
 * `Q8_0MatmulKernel`, mirroring the upstream `Q8_0MatmulMicrobenchTest`.
 * Provider selected by name — `scalar` or `panama`. Primary metric:
 * GOP/s for `2 * inputDim * outputDim` ops per invocation.
 */
internal class Q8MatmulScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-q8-matmul"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gops"
    override val unit: String = "gops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val inputDim: Int = if (smoke) 1024 else 4096
    private val outputDim: Int = if (smoke) 1024 else 4096
    override val parameters: Map<String, String> = mapOf(
        "input_dim" to inputDim.toString(),
        "output_dim" to outputDim.toString(),
        "kernel" to providerName,
    )

    private lateinit var kernel: Q8_0MatmulKernel
    private lateinit var input: FloatArray
    private lateinit var weight: ByteArray
    private lateinit var output: FloatArray

    private val blockSize: Int = 32
    private val bytesPerBlock: Int = 34

    override fun setup() {
        require(inputDim % blockSize == 0) { "inputDim must be a multiple of $blockSize, got $inputDim" }
        kernel = when (providerName.lowercase()) {
            "scalar" -> ScalarQ8_0MatmulKernel
            "panama", "panama-vector" -> PanamaVectorQ8_0MatmulKernel
            else -> error("unknown kernel provider: $providerName (use 'scalar' or 'panama')")
        }
        val blocksPerInputDim = inputDim / blockSize
        val numBlocks = blocksPerInputDim * outputDim
        val rng = Random((inputDim + outputDim).toLong())
        weight = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(weight)
        // Force the per-block scale `d` to a known small FP16 value so the
        // dequantized magnitudes stay finite across all blocks (matches
        // the convention used by the upstream Q8_0 microbench test).
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            weight[base] = 0x00.toByte()
            weight[base + 1] = 0x22.toByte()
        }
        input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        output = FloatArray(outputDim)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        kernel.matmul(input, 0, weight, 0, inputDim, outputDim, output, 0)
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = output[0]
        val ops = 2.0 * inputDim.toDouble() * outputDim.toDouble()
        val seconds = elapsedNs / 1_000_000_000.0
        return (ops / seconds) / 1e9
    }
}
