package sk.ainet.lang.tensor

import kotlinx.benchmark.*
import sk.ainet.lang.tensor.data.*
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
    public fun dequantQ4K(): FloatArray = (q4kData as PackedBlockStorage).toFloatArray()

    @Benchmark
    public fun dequantQ8_0(): FloatArray = (q80Data as PackedBlockStorage).toFloatArray()

    @Benchmark
    public fun dequantTernary(): FloatArray = (ternaryData as PackedBlockStorage).toFloatArray()
}

// --- BufferAccessor read performance ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class BufferAccessorBenchmark {
    private lateinit var accessor: ByteArrayAccessor
    private val readSize = 1024

    @Setup
    public fun setup() {
        val data = ByteArray(1024 * 1024) // 1 MB
        Random.nextBytes(data)
        accessor = ByteArrayAccessor(data)
    }

    @Benchmark
    public fun heapAccessor_readBytes_1KB(): ByteArray =
        accessor.readBytes(512_000, readSize)

    @Benchmark
    public fun heapAccessor_readByte_sequential(): Long {
        var sum = 0L
        for (i in 0 until readSize) {
            sum += accessor.readByte(i.toLong())
        }
        return sum
    }
}

// --- TensorData <-> TensorStorage bridge ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class StorageBridgeBenchmark {
    private lateinit var floatTd: DenseFloatArrayTensorData<FP32>
    private lateinit var q4kTd: Q4_KBlockTensorData
    private lateinit var floatStorage: TensorStorage
    private lateinit var q4kStorage: TensorStorage

    @Setup
    public fun setup() {
        floatTd = DenseFloatArrayTensorData(Shape(1024), FloatArray(1024) { it.toFloat() })
        q4kTd = Q4_KBlockTensorData.fromRawBytes(Shape(256), ByteArray(144))

        floatStorage = TensorStorageFactory.fromTensorData(floatTd)
        q4kStorage = TensorStorageFactory.fromTensorData(q4kTd)
    }

    @Benchmark
    public fun floatTensorData_toStorage(): TensorStorage =
        TensorStorageFactory.fromTensorData(floatTd)

    @Benchmark
    public fun q4kTensorData_toStorage(): TensorStorage =
        TensorStorageFactory.fromTensorData(q4kTd)

    @Benchmark
    public fun storage_toTensorData_float(): TensorData<FP32, Float> =
        TensorStorageFactory.toTensorData(floatStorage)

    @Benchmark
    public fun storage_toTensorData_q4k(): TensorData<DType, Byte> =
        TensorStorageFactory.toTensorData(q4kStorage)
}
