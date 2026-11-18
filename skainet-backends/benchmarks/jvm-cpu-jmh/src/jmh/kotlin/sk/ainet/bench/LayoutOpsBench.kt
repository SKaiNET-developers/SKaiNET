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
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * Layout and index ops on dense FP32 data (SKEEP-003, #1507): `transpose`, `permute`, and the
 * one-token embedding lookup through `gather` and `indexSelect`.
 *
 * These read their source array, and the lookups their indices, through the data's float or int
 * window. The lookup of one row is the small case where the cost of finding the window shows.
 *
 * ./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh -PjmhIncludes='LayoutOpsBench'
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
open class LayoutOpsBench {

    /** Columns of the `[64, width]` table. */
    @Param("16", "256")
    var width: Int = 16

    private lateinit var ops: TensorOps
    private lateinit var table: Tensor<FP32, Float>
    private lateinit var cube: Tensor<FP32, Float>
    private lateinit var oneToken: Tensor<DType, *>
    private val axes = intArrayOf(2, 0, 1)

    @Setup(Level.Trial)
    @Suppress("UNCHECKED_CAST")
    fun setup() {
        val ctx = DirectCpuExecutionContext()
        ops = ctx.ops
        table = ctx.fromFloatArray(Shape(64, width), FP32::class, FloatArray(64 * width) { (it % 13 - 6) * 0.25f })
        cube = ctx.fromFloatArray(Shape(4, 16, width), FP32::class, FloatArray(64 * width) { (it % 7 - 3) * 0.5f })
        oneToken = ctx.fromIntArray<Int32, Int>(Shape(1), Int32::class, intArrayOf(37)) as Tensor<DType, *>
    }

    @Benchmark
    fun transpose(): Any = ops.transpose(table)

    @Benchmark
    fun permute(): Any = ops.permute(cube, axes)

    @Benchmark
    fun gatherOneRow(): Any = ops.gather(table, oneToken, 0)

    @Benchmark
    fun indexSelectOneRow(): Any = ops.indexSelect(table, oneToken, 0)
}
