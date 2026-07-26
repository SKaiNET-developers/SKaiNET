package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.dense.DenseByteTensorArray
import sk.ainet.lang.tensor.storage.ActiveMemoryTracker
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int4
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Ternary
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Low level DenseTensorDataFactory factories.
 */

public class DenseTensorDataFactory: TensorDataFactory {

    private fun calcFlatIndex(shape: Shape, strides: IntArray, vararg indices: Int): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }

        var flatIndex = 0
        for (i in indices.indices) {
            require(indices[i] >= 0 && indices[i] < shape.dimensions[i]) {
                "Index ${indices[i]} out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += indices[i] * strides[i]
        }
        return flatIndex
    }


    @Suppress("UNCHECKED_CAST")
    public fun <T : DType> fromFloatArray(
        shape: Shape,
        data: FloatArray,
        dtype: T
    ): TensorData<T, Float> {
        return when (dtype) {
            is FP32, FP16 -> DenseFloatArrayTensorData<T>(shape, data.copyOf()) as TensorData<T, Float>
            else -> throw IllegalArgumentException("Unsupported dtype: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor filled with a constant value for the specified dtype.
     *
     * @param T the data type constraint extending DType
     * @param shape the shape of the tensor to create
     * @param value the constant value to fill the tensor with
     * @param dtype the data type instance
     * @return TensorData filled with the constant value
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType> full(shape: Shape, value: Number, dtype: T): TensorData<T, *> {
        return when (dtype) {
            is Int32 -> {
                val intValue = value.toInt()
                val data = IntArray(shape.volume) { intValue }
                createIntTensorData(shape, data) as TensorData<T, *>
            }

            is FP32, FP16 -> {
                val floatValue = value.toFloat()
                val data = FloatArray(shape.volume) { floatValue }
                createFloatTensorData(shape, data, dtype) as TensorData<T, *>
            }

            else -> throw IllegalArgumentException("Unsupported dtype for full: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor filled with random values from uniform distribution [0, 1) for floating point types
     * or [0, maxValue) for integer types.
     *
     * @param T the data type constraint extending DType
     * @param shape the shape of the tensor to create
     * @param dtype the data type instance
     * @param random the random number generator (optional, uses default if not provided)
     * @return TensorData filled with random values
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> randomInt(
        shape: Shape,
        dtype: T,
        random: Random,
        generator: (size: Int, random: Random) -> V
    ): TensorData<T, *> {
        return when (dtype) {
            is Int32 -> {
                val generatedData = generator(shape.volume, random)
                createIntTensorData(shape, generatedData as IntArray) as TensorData<T, *>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for random: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor filled with random values from normal distribution.
     *
     * @param T the data type constraint extending DType
     * @param shape the shape of the tensor to create
     * @param mean the mean of the normal distribution
     * @param std the standard deviation of the normal distribution
     * @param dtype the data type instance
     * @param random the random number generator (optional, uses default if not provided)
     * @return TensorData filled with normal distributed random values
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType> randn(
        shape: Shape,
        mean: Float = 0.0f,
        std: Float = 1.0f,
        dtype: T,
        random: Random = Random.Default
    ): TensorData<T, *> {
        return when (dtype) {
            is FP32, FP16 -> {
                val data = FloatArray(shape.volume) {
                    // Box-Muller transform for normal distribution
                    val u1 = random.nextFloat()
                    val u2 = random.nextFloat()
                    val z0 = sqrt(-2.0 * ln(u1.toDouble())).toFloat() *
                            cos(2.0 * PI * u2.toDouble()).toFloat()
                    mean + std * z0
                }
                createFloatTensorData(shape, data, dtype) as TensorData<T, *>
            }

            else -> throw IllegalArgumentException("Normal distribution only supported for floating point types: ${dtype.name}")
        }
    }

    // Helper methods to create tensor data instances

    private fun createIntTensorData(shape: Shape, data: IntArray): TensorData<Int32, Int> {
        ActiveMemoryTracker.recordCopy("DenseTensorDataFactory.createIntTensorData", data.size.toLong() * 4)
        return DenseIntArrayTensorData(shape, data.copyOf())
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> createFloatTensorData(shape: Shape, data: FloatArray, dtype: T): TensorData<T, Float> {
        ActiveMemoryTracker.recordCopy("DenseTensorDataFactory.createFloatTensorData", data.size.toLong() * 4)
        return DenseFloatArrayTensorData<T>(shape, data.copyOf()) as TensorData<T, Float>
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> createByteTensorData(shape: Shape, data: ByteArray): TensorData<T, Byte> {
        class ByteTensorDataImpl(
            private val denseArray: DenseByteTensorArray
        ) : TensorData<T, Byte>, ItemsAccessor<Byte> by denseArray {
            override val shape: Shape = denseArray.shape
        }
        val denseArray = DenseByteTensorArray(shape, data.copyOf())
        return ByteTensorDataImpl(denseArray) as TensorData<T, Byte>
    }

    // Convenience methods for scalar, vector, and matrix creation

    /**
     * Creates a scalar tensor from a single value.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> scalar(value: V): TensorData<T, V> {
        return when (value) {
            is Int -> {
                val data = intArrayOf(value)
                createIntTensorData(Shape(1), data) as TensorData<T, V>
            }

            is Float -> {
                val data = floatArrayOf(value)
                createFloatTensorData(Shape(1), data, FP32 as T) as TensorData<T, V>
            }

            is Byte -> {
                val data = intArrayOf(value.toInt())
                createIntTensorData(Shape(1), data) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("Unsupported value type: ${value!!}")
        }
    }

    /**
     * Creates a vector tensor from an array of values.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> vector(values: Array<V>): TensorData<T, V> {
        return when (values.firstOrNull()) {
            is Int -> {
                val data = values.map { (it as Int) }.toIntArray()
                createIntTensorData(Shape(values.size), data) as TensorData<T, V>
            }

            is Float -> {
                val data = values.map { (it as Float) }.toFloatArray()
                createFloatTensorData(Shape(values.size), data, FP32 as T) as TensorData<T, V>
            }

            is Byte -> {
                val data = values.map { (it as Byte).toInt() }.toIntArray()
                createIntTensorData(Shape(values.size), data) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("Unsupported value type: ${values.firstOrNull()?.let { it::class }}")
        }
    }

    /**
     * Creates a matrix tensor from multiple rows of values.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> matrix(vararg rows: Array<V>): TensorData<T, V> {
        require(rows.isNotEmpty()) { "Matrix must have at least one row" }
        val numRows = rows.size
        val numCols = rows[0].size
        require(rows.all { it.size == numCols }) { "All rows must have the same number of columns" }

        return when (rows[0].firstOrNull()) {
            is Int -> {
                val data = rows.flatMap { row -> row.map { (it as Int) } }.toIntArray()
                createIntTensorData(Shape(numRows, numCols), data) as TensorData<T, V>
            }

            is Float -> {
                val data = rows.flatMap { row -> row.map { (it as Float) } }.toFloatArray()
                createFloatTensorData(Shape(numRows, numCols), data, FP32 as T) as TensorData<T, V>
            }

            is Byte -> {
                val data = rows.flatMap { row -> row.map { (it as Byte).toInt() } }.toIntArray()
                createIntTensorData(Shape(numRows, numCols), data) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException(
                "Unsupported value type: ${
                    rows[0].firstOrNull()?.let { it::class }
                }"
            )
        }
    }

    /**
     * Creates a tensor from byte array data for various data types.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromByteArray(data: ByteArray, dtype: T): TensorData<T, V> {
        return when (dtype) {
            is FP32 -> {
                require(data.size % 4 == 0) { "Byte array size must be divisible by 4 for FP32" }
                val floats = FloatArray(data.size / 4) { i ->
                    val bytes = data.sliceArray(i * 4 until (i + 1) * 4)
                    // Convert little-endian bytes to float
                    val intBits = (bytes[3].toInt() and 0xFF shl 24) or
                            (bytes[2].toInt() and 0xFF shl 16) or
                            (bytes[1].toInt() and 0xFF shl 8) or
                            (bytes[0].toInt() and 0xFF)
                    Float.fromBits(intBits)
                }
                createFloatTensorData(Shape(floats.size), floats, dtype) as TensorData<T, V>
            }

            is Int8 -> {
                createByteTensorData<T>(Shape(data.size), data) as TensorData<T, V>
            }

            is Int4 -> {
                val values = mutableListOf<Byte>()
                for (byte in data) {
                    // Extract lower nibble (4 bits) and upper nibble
                    values.add((byte.toInt() and 0x0F).toByte()) // Lower nibble
                    values.add(((byte.toInt() and 0xF0) ushr 4).toByte()) // Upper nibble
                }
                createByteTensorData<T>(Shape(values.size), values.toByteArray()) as TensorData<T, V>
            }

            is Ternary -> {
                val values = mutableListOf<Byte>()
                for (byte in data) {
                    // Extract 2-bit values from byte (4 values per byte)
                    for (i in 0..3) {
                        val bits = (byte.toInt() shr (i * 2)) and 0x03
                        val value = when (bits) {
                            0b00 -> -1
                            0b01 -> 0
                            0b10 -> 1
                            else -> 0 // fallback for 0b11
                        }
                        values.add(value.toByte())
                    }
                }
                createByteTensorData<T>(Shape(values.size), values.toByteArray()) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("Unsupported dtype for fromByteArray: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor from float array with specified data type (overloaded version).
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromFloatArray(data: FloatArray, dtype: T): TensorData<T, V> {
        return when (dtype) {
            is FP32, is FP16, is BF16 -> {
                createFloatTensorData(Shape(data.size), data, dtype) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("fromFloatArray only supports floating point types: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor from int array with specified data type.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromIntArray(data: IntArray, dtype: T): TensorData<T, V> {
        return when (dtype) {
            is Int32 -> {
                createIntTensorData(Shape(data.size), data) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("fromIntArray only supports integer types: ${dtype.name}")
        }
    }

    override fun <T : DType, V> zeros(shape: Shape, dtype: KClass<T>): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> {
                val data = FloatArray(shape.volume) { 0.0f }
                createFloatTensorData(shape, data, FP32 as T) as TensorData<T, V>
            }
            FP16::class -> {
                val data = FloatArray(shape.volume) { 0.0f }
                createFloatTensorData(shape, data, FP16 as T) as TensorData<T, V>
            }
            BF16::class -> {
                // Float-backed, BF16-tagged (mirrors FP16). The dtype tag is what
                // the DSL trace / StableHLO export reads to emit `bf16` element
                // types — required for the Torq NPU (bf16-native weights).
                val data = FloatArray(shape.volume) { 0.0f }
                createFloatTensorData(shape, data, BF16 as T) as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { 0 }
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            Int8::class -> {
                val data = ByteArray(shape.volume) { 0 }
                createByteTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for zeros: $dtype")
        }
    }

    /**
     * Returns a [LazyZeroFloatArrayTensorData] / [LazyZeroIntArrayTensorData] for FP32 /
     * FP16 / Int32. The underlying primitive array materializes only on the first
     * `get`/`set`/`buffer` access. For Int8 (byte-backed) we currently fall back to
     * [zeros]; the eager byte allocation is rarely the dominant cost on real models.
     */
    override fun <T : DType, V> placeholder(shape: Shape, dtype: KClass<T>): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> LazyZeroFloatArrayTensorData<T>(shape) as TensorData<T, V>
            FP16::class -> LazyZeroFloatArrayTensorData<T>(shape) as TensorData<T, V>
            BF16::class -> LazyZeroFloatArrayTensorData<T>(shape) as TensorData<T, V>
            Int32::class -> LazyZeroIntArrayTensorData<T>(shape) as TensorData<T, V>
            else -> zeros(shape, dtype)
        }
    }

    /**
     * The [DType] instance for a 16-bit float tag. Both formats are stored float-backed here; only
     * the tag differs, and the tag is what the trace and StableHLO export read.
     */
    private fun narrowTag(dtype: KClass<*>): DType = if (dtype == BF16::class) BF16 else FP16

    override fun <T : DType, V> ones(shape: Shape, dtype: KClass<T>): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> {
                val data = FloatArray(shape.volume) { 1.0f }
                createFloatTensorData(shape, data, FP32 as T) as TensorData<T, V>
            }
            FP16::class, BF16::class -> {
                // Float-backed, narrow-tagged (mirrors zeros): the dtype tag is what the DSL
                // trace / StableHLO export reads to pick the MLIR element type.
                val data = FloatArray(shape.volume) { 1.0f }
                createFloatTensorData(shape, data, narrowTag(dtype) as T) as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { 1 }
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            Int8::class -> {
                val data = ByteArray(shape.volume) { 1 }
                createByteTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for ones: $dtype")
        }
    }

    override fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> {
                val floatValue = value.toFloat()
                val data = FloatArray(shape.volume) { floatValue }
                createFloatTensorData(shape, data, FP32 as T) as TensorData<T, V>
            }
            FP16::class, BF16::class -> {
                // Float-backed, narrow-tagged (mirrors zeros): the dtype tag is what the DSL
                // trace / StableHLO export reads to pick the MLIR element type.
                val floatValue = value.toFloat()
                val data = FloatArray(shape.volume) { floatValue }
                createFloatTensorData(shape, data, narrowTag(dtype) as T) as TensorData<T, V>
            }
            Int32::class -> {
                val intValue = value.toInt()
                val data = IntArray(shape.volume) { intValue }
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            Int8::class -> {
                val byteValue = value.toByte()
                val data = ByteArray(shape.volume) { byteValue }
                createByteTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for full: $dtype")
        }
    }

    override fun <T : DType, V> randn(
        shape: Shape,
        dtype: KClass<T>,
        mean: Float,
        std: Float,
        random: Random
    ): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> {
                val data = FloatArray(shape.volume)
                var hasSpare = false
                var spare = 0.0f

                for (i in data.indices) {
                    if (hasSpare) {
                        data[i] = spare * std + mean
                        hasSpare = false
                    } else {
                        val u1 = random.nextFloat()
                        val u2 = random.nextFloat()
                        val z0 = sqrt(-2.0 * ln(u1.toDouble())).toFloat() *
                                cos(2.0 * PI * u2.toDouble()).toFloat()
                        val z1 = sqrt(-2.0 * ln(u1.toDouble())).toFloat() *
                                kotlin.math.sin(2.0 * PI * u2.toDouble()).toFloat()

                        data[i] = z0 * std + mean
                        spare = z1
                        hasSpare = true
                    }
                }
                createFloatTensorData(shape, data, FP32 as T) as TensorData<T, V>
            }
            FP16::class, BF16::class -> {
                val data = FloatArray(shape.volume)
                var hasSpare = false
                var spare = 0.0f

                for (i in data.indices) {
                    if (hasSpare) {
                        data[i] = spare * std + mean
                        hasSpare = false
                    } else {
                        val u1 = random.nextFloat()
                        val u2 = random.nextFloat()
                        val z0 = sqrt(-2.0 * ln(u1.toDouble())).toFloat() *
                                cos(2.0 * PI * u2.toDouble()).toFloat()
                        val z1 = sqrt(-2.0 * ln(u1.toDouble())).toFloat() *
                                kotlin.math.sin(2.0 * PI * u2.toDouble()).toFloat()

                        data[i] = z0 * std + mean
                        spare = z1
                        hasSpare = true
                    }
                }
                createFloatTensorData(shape, data, narrowTag(dtype) as T) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("randn only supports floating point types: $dtype")
        }
    }

    override fun <T : DType, V> uniform(
        shape: Shape,
        dtype: KClass<T>,
        min: Float,
        max: Float,
        random: Random
    ): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> {
                val range = max - min
                val data = FloatArray(shape.volume) { random.nextFloat() * range + min }
                createFloatTensorData(shape, data, FP32 as T) as TensorData<T, V>
            }
            FP16::class -> {
                val range = max - min
                val data = FloatArray(shape.volume) { random.nextFloat() * range + min }
                createFloatTensorData(shape, data, FP16 as T) as TensorData<T, V>
            }
            Int32::class -> {
                val intMin = min.toInt()
                val intMax = max.toInt()
                val data = IntArray(shape.volume) { random.nextInt(intMin, intMax) }
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("uniform supports floating point and Int32 types: $dtype")
        }
    }

    override fun <T : DType, V> init(
        shape: Shape,
        dtype: KClass<T>,
        generator: (indices: IntArray) -> V
    ): TensorData<T, V> {
        val strides = IntArray(shape.dimensions.size)
        var stride = 1
        for (i in shape.dimensions.indices.reversed()) {
            strides[i] = stride
            stride *= shape.dimensions[i]
        }

        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> {
                val data = FloatArray(shape.volume) { flatIndex ->
                    val indices = IntArray(shape.dimensions.size)
                    var remaining = flatIndex
                    for (i in indices.indices) {
                        indices[i] = remaining / strides[i]
                        remaining %= strides[i]
                    }
                    (generator(indices) as Float)
                }
                createFloatTensorData(shape, data, FP32 as T) as TensorData<T, V>
            }
            FP16::class -> {
                val data = FloatArray(shape.volume) { flatIndex ->
                    val indices = IntArray(shape.dimensions.size)
                    var remaining = flatIndex
                    for (i in indices.indices) {
                        indices[i] = remaining / strides[i]
                        remaining %= strides[i]
                    }
                    (generator(indices) as Float)
                }
                createFloatTensorData(shape, data, FP16 as T) as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { flatIndex ->
                    val indices = IntArray(shape.dimensions.size)
                    var remaining = flatIndex
                    for (i in indices.indices) {
                        indices[i] = remaining / strides[i]
                        remaining %= strides[i]
                    }
                    (generator(indices) as Int)
                }
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            Int8::class -> {
                // Store Int8 tensors with Int-backed storage for arithmetic operations
                val data = IntArray(shape.volume) { flatIndex ->
                    val indices = IntArray(shape.dimensions.size)
                    var remaining = flatIndex
                    for (i in indices.indices) {
                        indices[i] = remaining / strides[i]
                        remaining %= strides[i]
                    }
                    (generator(indices) as Int)
                }
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("init supports floating point, Int32, and Int8 types: $dtype")
        }
    }

    override fun <T : DType, V> randomInit(
        shape: Shape,
        dtype: KClass<T>,
        generator: (random: Random) -> V,
        random: Random
    ): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> {
                val data = FloatArray(shape.volume) { (generator(random) as Float) }
                createFloatTensorData(shape, data, FP32 as T) as TensorData<T, V>
            }
            FP16::class -> {
                val data = FloatArray(shape.volume) { (generator(random) as Float) }
                createFloatTensorData(shape, data, FP16 as T) as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { (generator(random) as Int) }
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            Int8::class -> {
                val data = ByteArray(shape.volume) { (generator(random) as Byte) }
                createByteTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("randomInit supports floating point, Int32, and Int8 types: $dtype")
        }
    }

    override fun <T : DType, V> fromFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> {
                createFloatTensorData(shape, data, FP32 as T) as TensorData<T, V>
            }
            FP16::class, BF16::class -> {
                createFloatTensorData(shape, data, narrowTag(dtype) as T) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("fromFloatArray only supports floating point types: $dtype")
        }
    }

    override fun <T : DType, V> fromIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            Int32::class -> {
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            Int8::class -> {
                // For Int8 with IntArray input, store as Int-backed TensorData to allow Int operations
                createIntTensorData(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("fromIntArray only supports Int32/Int8 types: $dtype")
        }
    }

    override fun <T : DType, V> fromByteArray(
        shape: Shape,
        dtype: KClass<T>,
        data: ByteArray
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            Int8::class -> {
                createByteTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("fromByteArray only supports Int8 types with shape: $dtype")
        }
    }

    // --- Zero-copy wrap methods (borrow semantics) ---

    override fun <T : DType, V> wrapFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            FP32::class -> DenseFloatArrayTensorData<T>(shape, data) as TensorData<T, V>
            FP16::class -> DenseFloatArrayTensorData<T>(shape, data) as TensorData<T, V>
            else -> throw IllegalArgumentException("wrapFloatArray only supports floating point types: $dtype")
        }
    }

    override fun <T : DType, V> wrapIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            Int32::class -> DenseIntArrayTensorData<T>(shape, data) as TensorData<T, V>
            else -> throw IllegalArgumentException("wrapIntArray only supports Int32 types: $dtype")
        }
    }

    override fun <T : DType, V> wrapByteArray(
        shape: Shape,
        dtype: KClass<T>,
        data: ByteArray
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            Int8::class -> {
                val denseArray = DenseByteTensorArray(shape, data)
                class WrappedByteTensorData(
                    private val inner: DenseByteTensorArray
                ) : TensorData<T, Byte>, ItemsAccessor<Byte> by inner {
                    override val shape: Shape = inner.shape
                }
                WrappedByteTensorData(denseArray) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("wrapByteArray only supports Int8 types: $dtype")
        }
    }
}
