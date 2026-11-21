package sk.ainet.lang.tensor

import kotlinx.benchmark.*
import sk.ainet.lang.tensor.data.*
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.storage.*
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.random.Random

/**
 * JMH benchmarks for the memory-first storage layer.
 *
 * Run: ./gradlew :skainet-lang:skainet-lang-core:jvmBenchmark
 */

// --- Array creation: borrowed (wrap) vs copied (from) ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class ArrayCreationBenchmark {
    private val factory = DenseTensorDataFactory()
    private val shape = Shape(1024, 1024) // 1M elements
    private lateinit var floatData: FloatArray

    @Setup
    public fun setup() {
        floatData = FloatArray(1024 * 1024) { Random.nextFloat() }
    }

    @Benchmark
    public fun wrapFloatArray_zeroCopy(): TensorData<FP32, Float> =
        factory.wrapFloatArray(shape, FP32::class, floatData)

    @Benchmark
    public fun fromFloatArray_copy(): TensorData<FP32, Float> =
        factory.fromFloatArray(shape, FP32::class, floatData)
}

// --- Dequantization throughput ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class DequantizationBenchmark {
    private lateinit var q4kData: Q4_KBlockTensorData
    private lateinit var q80Data: Q8_0BlockTensorData
    private lateinit var ternaryData: Ternary2BitTensorData

    @Setup
    public fun setup() {
        // Q4_K: 100 blocks = 25600 elements
        val q4kBytes = ByteArray(100 * Q4_KTensorData.BYTES_PER_BLOCK)
        Random.nextBytes(q4kBytes)
        q4kData = Q4_KBlockTensorData.fromRawBytes(Shape(25600), q4kBytes)

        // Q8_0: 800 blocks = 25600 elements
        val q80Bytes = ByteArray(800 * Q8_0TensorData.BYTES_PER_BLOCK)
        Random.nextBytes(q80Bytes)
        q80Data = Q8_0BlockTensorData.fromRawBytes(Shape(25600), q80Bytes)

        // Ternary: 25600 elements = 6400 packed bytes
        val ternaryBytes = ByteArray(6400)
        Random.nextBytes(ternaryBytes)
        ternaryData = Ternary2BitTensorData(Shape(25600), ternaryBytes)
    }

    @Benchmark
    public fun dequantQ4K(): FloatArray = q4kData.toFloatArray()

    @Benchmark
    public fun dequantQ8_0(): FloatArray = q80Data.toFloatArray()

    @Benchmark
    public fun dequantTernary(): FloatArray = ternaryData.toFloatArray()
}

// --- TensorData <-> TensorView ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class StorageViewBenchmark {
    private lateinit var floatTd: DenseFloatArrayTensorData<FP32>
    private lateinit var q4kTd: Q4_KBlockTensorData
    private lateinit var q4kBytes: ByteArray
    private lateinit var q4kView: TensorView

    @Setup
    public fun setup() {
        floatTd = DenseFloatArrayTensorData(Shape(1024), FloatArray(1024) { it.toFloat() })
        q4kBytes = ByteArray(144)
        q4kTd = Q4_KBlockTensorData.fromRawBytes(Shape(256), q4kBytes)
        q4kView = TensorView.ofBytes(Storage.Heap.wrap(q4kBytes, mutable = false), Shape(256), Format(FP32, TensorEncoding.Q4_K))
    }

    /** The view a dense data object gives for its own array. */
    @Benchmark
    public fun floatTensorData_view(): TensorView? = floatTd.view

    /** The view a packed data object gives for its own bytes. */
    @Benchmark
    public fun q4kTensorData_view(): TensorView? = q4kTd.view

    /** A view over raw file bytes, as a streaming reader builds it. */
    @Benchmark
    public fun q4kBytes_toView(): TensorView =
        TensorView.ofBytes(Storage.Heap.wrap(q4kBytes, mutable = false), Shape(256), Format(FP32, TensorEncoding.Q4_K))

    /** One block decoded through the view (the reference path, not a kernel). */
    @Benchmark
    public fun q4kView_toFloatArray(): FloatArray = q4kView.toFloatArray()
}
