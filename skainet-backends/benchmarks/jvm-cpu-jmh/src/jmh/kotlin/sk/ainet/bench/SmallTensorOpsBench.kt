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
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.FP32

/**
 * Per-call overhead of the dense fast paths on small tensors (SKEEP-003, #1507).
 *
 * The ops find a tensor's floats through its view. On a million elements that lookup is invisible;
 * on a few dozen it is a visible share of the call, and decode loops make many such calls (#949).
 * `ElementwiseAdd1MBench` cannot see it, so this one measures the small sizes.
 *
 * ./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh -PjmhIncludes='SmallTensorOpsBench'
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
open class SmallTensorOpsBench {

    @Param("16", "256", "4096")
    var size: Int = 16

    private lateinit var ops: TensorOps
    private lateinit var a: Tensor<FP32, Float>
    private lateinit var b: Tensor<FP32, Float>
    private lateinit var row: Tensor<FP32, Float>
    private lateinit var weight: Tensor<FP32, Float>

    @Setup(Level.Trial)
    fun setup() {
        val ctx = DirectCpuExecutionContext()
        ops = ctx.ops
        a = ctx.fromFloatArray(Shape(size), FP32::class, FloatArray(size) { (it % 13 - 6) * 0.25f })
        b = ctx.fromFloatArray(Shape(size), FP32::class, FloatArray(size) { (it % 7 - 3) * 0.5f })
        // A decode-step projection: one row of `size` against a [size, 16] weight.
        row = ctx.fromFloatArray(Shape(1, size), FP32::class, FloatArray(size) { (it % 11 - 5) * 0.125f })
        weight = ctx.fromFloatArray(Shape(size, 16), FP32::class, FloatArray(size * 16) { (it % 17 - 8) * 0.0625f })
    }

    @Benchmark
    fun add(): Any = ops.add(a, b)

    @Benchmark
    fun multiply(): Any = ops.multiply(a, b)

    @Benchmark
    fun relu(): Any = ops.relu(a)

    @Benchmark
    fun matmulRow(): Any = ops.matmul(row, weight)
}
