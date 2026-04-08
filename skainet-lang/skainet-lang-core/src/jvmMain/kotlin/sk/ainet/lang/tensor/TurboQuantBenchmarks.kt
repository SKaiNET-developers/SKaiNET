package sk.ainet.lang.tensor

import kotlinx.benchmark.*
import sk.ainet.lang.tensor.ops.turboquant.*
import sk.ainet.lang.tensor.storage.*
import kotlin.random.Random

/**
 * JMH benchmarks for TurboQuant KV-cache compression.
 *
 * Measures encode/decode throughput, compression ratio, and accuracy
 * for different TurboQuant configurations.
 *
 * Run: ./gradlew :skainet-lang:skainet-lang-core:jvmBenchmark
 */

// --- Encode throughput ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class TurboQuantEncodeBenchmark {
    private lateinit var vector128: FloatArray
    private lateinit var vector256: FloatArray
    private lateinit var vector512: FloatArray
    private lateinit var config4Bit: TurboQuantConfig
    private lateinit var config3Bit: TurboQuantConfig
    private lateinit var config8Bit: TurboQuantConfig
    private lateinit var configQjl: TurboQuantConfig

    @Setup
    public fun setup() {
        val rng = Random(42)
        vector128 = FloatArray(128) { rng.nextFloat() * 2 - 1 }
        vector256 = FloatArray(256) { rng.nextFloat() * 2 - 1 }
        vector512 = FloatArray(512) { rng.nextFloat() * 2 - 1 }
        config4Bit = TurboQuantConfig.polarOnly(bits = 4, seed = 42)
        config3Bit = TurboQuantConfig.polarOnly(bits = 3, seed = 42)
        config8Bit = TurboQuantConfig.polarOnly(bits = 8, seed = 42)
        configQjl = TurboQuantConfig.polarPlusQjl(bits = 4, residualBits = 1, seed = 42)
    }

    @Benchmark
    public fun encode_4bit_128d(): TurboQuantBlock =
        TurboQuantCodec.encode(vector128, config4Bit)

    @Benchmark
    public fun encode_4bit_256d(): TurboQuantBlock =
        TurboQuantCodec.encode(vector256, config4Bit)

    @Benchmark
    public fun encode_3bit_128d(): TurboQuantBlock =
        TurboQuantCodec.encode(vector128, config3Bit)

    @Benchmark
    public fun encode_8bit_128d(): TurboQuantBlock =
        TurboQuantCodec.encode(vector128, config8Bit)

    @Benchmark
    public fun encode_4bit_qjl_128d(): TurboQuantBlock =
        TurboQuantCodec.encode(vector128, configQjl)
}

// --- Decode throughput ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class TurboQuantDecodeBenchmark {
    private lateinit var block4Bit128: TurboQuantBlock
    private lateinit var block4Bit256: TurboQuantBlock
    private lateinit var block3Bit128: TurboQuantBlock
    private lateinit var block8Bit128: TurboQuantBlock
    private lateinit var blockQjl128: TurboQuantBlock

    @Setup
    public fun setup() {
        val rng = Random(42)
        val v128 = FloatArray(128) { rng.nextFloat() * 2 - 1 }
        val v256 = FloatArray(256) { rng.nextFloat() * 2 - 1 }

        block4Bit128 = TurboQuantCodec.encode(v128, TurboQuantConfig.polarOnly(bits = 4, seed = 42))
        block4Bit256 = TurboQuantCodec.encode(v256, TurboQuantConfig.polarOnly(bits = 4, seed = 42))
        block3Bit128 = TurboQuantCodec.encode(v128, TurboQuantConfig.polarOnly(bits = 3, seed = 42))
        block8Bit128 = TurboQuantCodec.encode(v128, TurboQuantConfig.polarOnly(bits = 8, seed = 42))
        blockQjl128 = TurboQuantCodec.encode(v128, TurboQuantConfig.polarPlusQjl(bits = 4, seed = 42))
    }

    @Benchmark
    public fun decode_4bit_128d(): FloatArray =
        TurboQuantCodec.decode(block4Bit128)

    @Benchmark
    public fun decode_4bit_256d(): FloatArray =
        TurboQuantCodec.decode(block4Bit256)

    @Benchmark
    public fun decode_3bit_128d(): FloatArray =
        TurboQuantCodec.decode(block3Bit128)

    @Benchmark
    public fun decode_8bit_128d(): FloatArray =
        TurboQuantCodec.decode(block8Bit128)

    @Benchmark
    public fun decode_4bit_qjl_128d(): FloatArray =
        TurboQuantCodec.decode(blockQjl128)
}

// --- Bit-packing throughput ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class BitPackerBenchmark {
    private lateinit var codes128: ByteArray
    private lateinit var codes1024: ByteArray
    private lateinit var packed4Bit: ByteArray
    private lateinit var packed2Bit: ByteArray

    @Setup
    public fun setup() {
        codes128 = ByteArray(128) { (it % 7 - 3).toByte() }
        codes1024 = ByteArray(1024) { (it % 7 - 3).toByte() }
        packed4Bit = BitPacker.pack(codes1024, 4)
        packed2Bit = BitPacker.pack(ByteArray(1024) { (it % 3 - 1).toByte() }, 2)
    }

    @Benchmark
    public fun pack_4bit_128(): ByteArray = BitPacker.pack(codes128, 4)

    @Benchmark
    public fun pack_4bit_1024(): ByteArray = BitPacker.pack(codes1024, 4)

    @Benchmark
    public fun unpack_4bit_1024(): ByteArray = BitPacker.unpack(packed4Bit, 1024, 4)

    @Benchmark
    public fun pack_2bit_1024(): ByteArray = BitPacker.pack(ByteArray(1024) { (it % 3 - 1).toByte() }, 2)

    @Benchmark
    public fun unpack_2bit_1024(): ByteArray = BitPacker.unpack(packed2Bit, 1024, 2)
}

// --- Random rotation throughput ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class RandomRotationBenchmark {
    private lateinit var vector128: FloatArray
    private lateinit var vector256: FloatArray

    @Setup
    public fun setup() {
        val rng = Random(42)
        vector128 = FloatArray(128) { rng.nextFloat() * 2 - 1 }
        vector256 = FloatArray(256) { rng.nextFloat() * 2 - 1 }
    }

    @Benchmark
    public fun rotate_128d(): FloatArray {
        val v = vector128.copyOf()
        RandomRotation.rotate(v, 42)
        return v
    }

    @Benchmark
    public fun rotate_256d(): FloatArray {
        val v = vector256.copyOf()
        RandomRotation.rotate(v, 42)
        return v
    }

    @Benchmark
    public fun rotateInverse_128d(): FloatArray {
        val v = vector128.copyOf()
        RandomRotation.rotate(v, 42)
        RandomRotation.inverseRotate(v, 42)
        return v
    }
}

// --- KV cache throughput ---

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
public open class TurboQuantKvCacheBenchmark {
    private lateinit var denseStore: DefaultKvCacheStore
    private lateinit var turboStore: TurboQuantKvCacheStore
    private lateinit var keyProjection: FloatArray
    private lateinit var valueProjection: FloatArray

    @Setup
    public fun setup() {
        val rng = Random(42)
        val numHeads = 8
        val headDim = 128
        val maxSeqLen = 256

        denseStore = DefaultKvCacheStore(
            KvCacheConfig(numLayers = 1, numHeads = numHeads, headDim = headDim, maxSeqLen = maxSeqLen)
        )
        turboStore = TurboQuantKvCacheStore(
            KvCacheConfig(
                numLayers = 1, numHeads = numHeads, headDim = headDim, maxSeqLen = maxSeqLen,
                keyEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 4),
                valueEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 4)
            ),
            keyConfig = TurboQuantConfig.polarOnly(bits = 4),
            valueConfig = TurboQuantConfig.polarOnly(bits = 4)
        )

        keyProjection = FloatArray(numHeads * headDim) { rng.nextFloat() * 2 - 1 }
        valueProjection = FloatArray(numHeads * headDim) { rng.nextFloat() * 2 - 1 }
    }

    @Benchmark
    public fun appendToken_dense() {
        denseStore.clear()
        denseStore.appendToken(0, keyProjection, valueProjection)
    }

    @Benchmark
    public fun appendToken_turbo4bit() {
        turboStore.clear()
        turboStore.appendToken(0, keyProjection, valueProjection)
    }

    @Benchmark
    public fun readKeys_dense_16tokens() {
        denseStore.clear()
        for (i in 0 until 16) denseStore.appendToken(0, keyProjection, valueProjection)
        denseStore.readKeys(0)
    }

    @Benchmark
    public fun readKeys_turbo4bit_16tokens() {
        turboStore.clear()
        for (i in 0 until 16) turboStore.appendToken(0, keyProjection, valueProjection)
        turboStore.readKeys(0)
    }
}
