package sk.ainet.bench

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Level
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OutputTimeUnit
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.exec.kernel.PanamaVectorMatmulKernel
import sk.ainet.exec.kernel.ScalarMatmulKernel

/**
 * Direct kernel-level matmul bench: `Fp32MatmulKernel.matmul` only,
 * with no `TensorOps` wrapper / dispatch / context allocation in the
 * timed region. Used to validate the M5 milestone target — Panama
 * Vector kernel ≥ 1.5× scalar — independent of the rest of the op
 * pipeline.
 *
 * Compare against `MatmulBench`, which exercises the same operation
 * through `ctx.ops.matmul` (production routing). Until
 * `DefaultCpuOpsJvm.matmul` is wired through `KernelRegistry`, only
 * this bench reflects pure kernel-vs-kernel performance.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class KernelMatmulBench {

    @Param("256", "512", "1024")
    var size: Int = 512

    @Param("scalar", "panama")
    var provider: String = "panama"

    private lateinit var kernel: Fp32MatmulKernel
    private lateinit var a: FloatArray
    private lateinit var b: FloatArray
    private lateinit var out: FloatArray

    @Setup(Level.Trial)
    fun setup() {
        kernel = when (provider) {
            "scalar" -> ScalarMatmulKernel
            "panama" -> PanamaVectorMatmulKernel
            else -> error("unknown provider: $provider")
        }
        val n = size
        // Same input seeding as MatmulBench so numbers compare cleanly.
        a = FloatArray(n * n) { ((it % 251) - 125).toFloat() / 127f }
        b = FloatArray(n * n) { ((it * 13 % 257) - 128).toFloat() / 127f }
        out = FloatArray(n * n)
    }

    @Benchmark
    fun matmul_fp32_square(): FloatArray {
        val n = size
        kernel.matmul(
            a, 0, n,
            b, 0, n,
            out, 0, n,
            n, n, n,
        )
        return out
    }
}
