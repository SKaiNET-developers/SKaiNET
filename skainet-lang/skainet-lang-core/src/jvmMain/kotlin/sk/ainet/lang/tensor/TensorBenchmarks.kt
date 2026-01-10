package sk.ainet.lang.tensor

import kotlinx.benchmark.*
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.context.DefaultDataExecutionContext
import kotlin.random.Random

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class TensorCreationBenchmark {
    private val factory: DenseTensorDataFactory = DenseTensorDataFactory()
    private val smallShape: Shape = Shape(16, 16)
    private val mediumShape: Shape = Shape(256, 256)
    private val largeShape: Shape = Shape(1024, 1024)

    @Benchmark
    public fun createSmallZerosFP32(): TensorData<FP32, Float> = factory.zeros(smallShape, FP32::class)

    @Benchmark
    public fun createMediumZerosFP32(): TensorData<FP32, Float> = factory.zeros(mediumShape, FP32::class)

    @Benchmark
    public fun createLargeZerosFP32(): TensorData<FP32, Float> = factory.zeros(largeShape, FP32::class)

    @Benchmark
    public fun createSmallZerosInt32(): TensorData<Int32, Int> = factory.zeros(smallShape, Int32::class)

    @Benchmark
    public fun createMediumZerosInt32(): TensorData<Int32, Int> = factory.zeros(mediumShape, Int32::class)

    @Benchmark
    public fun createLargeZerosInt32(): TensorData<Int32, Int> = factory.zeros(largeShape, Int32::class)

    @Benchmark
    public fun createMediumRandnFP32(): TensorData<FP32, Float> = factory.randn(mediumShape, FP32::class, 0.0f, 1.0f, Random.Default)
}

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class CosineDistanceBenchmark {
    private val ctx: DefaultDataExecutionContext = DefaultDataExecutionContext()
    private lateinit var a: Tensor<FP32, Float>
    private lateinit var b: Tensor<FP32, Float>

    @Setup
    public fun setup() {
        val shape = Shape(1024)
        a = ctx.zeros(shape, FP32::class)
        b = ctx.zeros(shape, FP32::class)
    }

    @Benchmark
    public fun cosineDistance(): Tensor<FP32, Float> = a.cosineDistance(b)
}
