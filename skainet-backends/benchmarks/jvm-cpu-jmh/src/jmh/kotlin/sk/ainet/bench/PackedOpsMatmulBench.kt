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
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.PanamaVectorQ4KMatmulKernel
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.FP32

/**
 * What the ops layer adds on top of the packed Q4_K kernel (SKEEP-003, #1507): the kernel is
 * chosen by `KernelKey` from the weight's view, so this is the price of that lookup.
 *
 * `kernel` is the bare Panama kernel, as `QuantizedMatmulBench` measures it. `opsLabel` is
 * `ops.matmul(x, w)` with the weight as the `[in, out]` label over feed-order bytes.
 * `opsWeightTransposed` is `ops.matmulWeightTransposed(x, w)` with a canonical `[out, in]` weight,
 * which is relayouted once and cached. The ops run under the hardware schedule.
 *
 * ./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh -PjmhIncludes='PackedOpsMatmulBench'
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class PackedOpsMatmulBench {

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
    private lateinit var label: Tensor<FP32, Float>
    private lateinit var canonical: Tensor<FP32, Float>

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
        @Suppress("UNCHECKED_CAST")
        label = ctx.fromData(Q4_KBlockTensorData(Shape(inputDim, outputDim), packed) as TensorData<FP32, Float>, FP32::class)
        @Suppress("UNCHECKED_CAST")
        canonical = ctx.fromData(Q4_KBlockTensorData(Shape(outputDim, inputDim), packed.copyOf()) as TensorData<FP32, Float>, FP32::class)
    }

    @Benchmark
    fun kernel(): FloatArray {
        for (b in 0 until batch) {
            PanamaVectorQ4KMatmulKernel.matmul(input, b * inputDim, packed, 0, inputDim, outputDim, output, b * outputDim)
        }
        return output
    }

    @Benchmark
    fun opsLabel(): Tensor<FP32, Float> = ops.matmul(x, label)

    @Benchmark
    fun opsWeightTransposed(): Tensor<FP32, Float> = ops.matmulWeightTransposed(x, canonical)
}
