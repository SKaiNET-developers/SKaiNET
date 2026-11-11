package sk.ainet.bench.spike

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
import org.openjdk.jmh.annotations.TearDown
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.exec.kernel.PanamaVectorQ4KMatmulKernel
import sk.ainet.exec.tensor.ops.Q4KKeyDispatchSpike
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.FP32

/**
 * SKEEP-003 G4 spike (#1507): what does it cost to pick the Q4_K matmul kernel by `KernelKey`
 * (the weight's Format and layout class) instead of by `is Q4_KTensorData`?
 *
 * Same weight, same activation, same Panama kernel underneath. `kernel` is the bare kernel call
 * (what `QuantizedMatmulBench` measures); `ladder` is `ops.matmul` as it is today; `key` is
 * `ops.matmul` with the spike path on. Both ops variants run with the hardware schedule, which is
 * what the kernel's schedule-less entry point uses and the only one a view kernel can get today.
 *
 * ./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh -PjmhIncludes='Q4KDispatchSpikeBench'
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class Q4KDispatchSpikeBench {

    @Param("1024-1024", "4096-4096")
    var shape: String = "4096-4096"

    @Param("1", "8")
    var batch: Int = 1

    private var inputDim = 0
    private var outputDim = 0
    private lateinit var input: FloatArray
    private lateinit var packed: ByteArray
    private lateinit var output: FloatArray
    private lateinit var ops: TensorOps
    private lateinit var x: Tensor<FP32, Float>
    private lateinit var w: Tensor<FP32, Float>

    @Setup(Level.Trial)
    fun setup() {
        val parts = shape.split("-")
        inputDim = parts[0].toInt()
        outputDim = parts[1].toInt()
        val numBlocks = (inputDim / 256) * outputDim
        packed = ByteArray(numBlocks * 144)
        Random(42).nextBytes(packed)
        for (block in 0 until numBlocks) {
            val base = block * 144
            packed[base] = 0x00; packed[base + 1] = 0x3C
            packed[base + 2] = 0x00; packed[base + 3] = 0x3C
        }
        input = FloatArray(batch * inputDim) { ((it % 251) - 125).toFloat() / 127f }
        output = FloatArray(batch * outputDim)

        val ctx = DirectCpuExecutionContext().withSchedule(CoroutineSchedule.hardware())
        ops = ctx.ops
        x = ctx.fromFloatArray(Shape(batch, inputDim), FP32::class, input)
        // What `matmul` receives for a packed weight today: `[in, out]` data over feed-order bytes.
        @Suppress("UNCHECKED_CAST")
        w = ctx.fromData(Q4_KBlockTensorData(Shape(inputDim, outputDim), packed) as TensorData<FP32, Float>, FP32::class)
    }

    @TearDown(Level.Trial)
    fun tearDown() { Q4KKeyDispatchSpike.enabled = false }

    @Benchmark
    fun kernel(): FloatArray {
        for (b in 0 until batch) {
            PanamaVectorQ4KMatmulKernel.matmul(input, b * inputDim, packed, 0, inputDim, outputDim, output, b * outputDim)
        }
        return output
    }

    @Benchmark
    fun ladder(): Tensor<FP32, Float> {
        Q4KKeyDispatchSpike.enabled = false
        return ops.matmul(x, w)
    }

    @Benchmark
    fun key(): Tensor<FP32, Float> {
        Q4KKeyDispatchSpike.enabled = true
        return ops.matmul(x, w)
    }
}
