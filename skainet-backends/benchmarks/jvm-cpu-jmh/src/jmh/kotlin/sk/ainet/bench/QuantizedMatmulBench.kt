package sk.ainet.bench

import java.util.concurrent.TimeUnit
import kotlin.random.Random
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Level
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OutputTimeUnit
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import sk.ainet.exec.kernel.PanamaVectorQ4KMatmulKernel

/**
 * F32-input × Q4_K-weight matmul bench: measures the SIMD-fused
 * Panama kernel ([PanamaVectorQ4KMatmulKernel]) at typical LLM matmul
 * shapes for Gemma 4 E2B Q4_K_M:
 *   - 1024 x 1024 — small attention projection
 *   - 4096 x 4096 — hidden→hidden / FFN gate
 *   - 4096 x 1024 — hidden→KV slice
 *
 * Each `inputDim` must be a multiple of 256 (Q4_K block size). Packed
 * layout is input-block-major (`(blockIdx * outputDim + o) * 144`).
 *
 * Direct comparison vs the prior `JvmQuantizedVectorKernels.matmulQ4_KVec`
 * partial-vec implementation is via the parity test in
 * `PanamaVectorQ4KMatmulKernelTest`, which exercises both code paths.
 * The internal visibility of that legacy kernel keeps it out of the
 * cross-module bench harness.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class QuantizedMatmulBench {

    @Param("1024-1024", "4096-1024", "4096-4096")
    var shape: String = "4096-4096"

    private var inputDim: Int = 0
    private var outputDim: Int = 0
    private lateinit var input: FloatArray
    private lateinit var packedWeights: ByteArray
    private lateinit var output: FloatArray

    @Setup(Level.Trial)
    fun setup() {
        val parts = shape.split("-")
        inputDim = parts[0].toInt()
        outputDim = parts[1].toInt()
        require(inputDim % 256 == 0) { "inputDim must be multiple of 256, got $inputDim" }

        val numBlocks = (inputDim / 256) * outputDim
        val rng = Random(42)
        packedWeights = ByteArray(numBlocks * 144)
        rng.nextBytes(packedWeights)
        // Force d / dMin per block to 1.0f16 (0x3C00) so dequantized
        // magnitudes stay within finite range for steady-state runs.
        for (block in 0 until numBlocks) {
            val base = block * 144
            packedWeights[base] = 0x00.toByte(); packedWeights[base + 1] = 0x3C.toByte()
            packedWeights[base + 2] = 0x00.toByte(); packedWeights[base + 3] = 0x3C.toByte()
        }
        input = FloatArray(inputDim) { ((it % 251) - 125).toFloat() / 127f }
        output = FloatArray(outputDim)
    }

    @Benchmark
    fun matmul_q4k_panama(): FloatArray {
        PanamaVectorQ4KMatmulKernel.matmul(
            input, 0,
            packedWeights, 0,
            inputDim, outputDim,
            output, 0,
        )
        return output
    }
}
