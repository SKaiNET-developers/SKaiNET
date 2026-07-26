package sk.ainet.io.safetensors

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.DataType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import kotlin.math.pow
import kotlin.reflect.KClass

/**
 * ParametersLoader implementation for SafeTensors format using streaming.
 *
 * Uses [StreamingSafeTensorsReader] for memory-efficient loading - only
 * parses the header (~1KB-1MB) and loads tensors on-demand.
 *
 * Supported conversions:
 * - F32/F64 tensors -> FP32 (F64 downcast with warning)
 * - I32/I64 tensors -> Int32 (I64 downcast with warning)
 * - I8/U8 tensors -> Int8
 * - F16 tensors -> FP32 (with dequantization)
 * - BF16 tensors -> FP32 (default) OR native BF16 storage (`bf16Policy = KEEP_NATIVE`)
 *
 * Where possible, decoded arrays are wrapped (borrowed) rather than copied
 * into TensorData, avoiding a second allocation. The raw-byte decode step
 * (little-endian bytes → typed array) is still necessary.
 *
 * @param sourceProvider Factory providing RandomAccessSource to the SafeTensors file
 * @param onProgress Optional progress callback (current, total, tensorName)
 * @param bf16Policy How to handle `BFLOAT16` tensors. Default is
 *   [Bf16LoadPolicy.DEQUANT_TO_FP32] — backward-compatible with all
 *   existing consumers. Flip to [Bf16LoadPolicy.KEEP_NATIVE] to keep
 *   weights in their on-disk BF16 layout and let the matmul dispatch
 *   route to a vectorised BF16 kernel.
 */
class SafeTensorsParametersLoader(
    private val sourceProvider: () -> RandomAccessSource,
    private val onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
    private val bf16Policy: Bf16LoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
    private val fp16Policy: NarrowFloatLoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
) : ParametersLoader {

    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        StreamingSafeTensorsReader.open(sourceProvider()).use { reader ->
            val tensors = reader.tensors
            val total = tensors.size.toLong()
            var current = 0L

            for (tensorInfo in tensors) {
                val bytes = reader.loadTensorData(tensorInfo)
                val shape = Shape(*tensorInfo.shape.map { it.toInt() }.toIntArray())

                @Suppress("UNCHECKED_CAST")
                val tensor: Tensor<T, V> = when (tensorInfo.dataType) {
                    DataType.FLOAT32 -> {
                        require(dtype == FP32::class) {
                            "SafeTensors F32 tensor '${tensorInfo.name}' requires FP32 dtype, got ${dtype.simpleName}"
                        }
                        val floats = bytesToFloatArray(bytes)
                        // Wrap the decoded array (zero-copy) — it was freshly allocated by bytesToFloatArray
                        ctx.wrapFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
                    }

                    DataType.FLOAT64 -> {
                        require(dtype == FP32::class) {
                            "SafeTensors F64 tensor '${tensorInfo.name}' requires FP32 dtype (downcast), got ${dtype.simpleName}"
                        }
                        println("WARNING: Downcasting F64 tensor '${tensorInfo.name}' to F32")
                        val doubles = bytesToDoubleArray(bytes)
                        val floats = FloatArray(doubles.size) { doubles[it].toFloat() }
                        ctx.wrapFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
                    }

                    DataType.FLOAT16 -> {
                        require(dtype == FP32::class) {
                            "SafeTensors F16 tensor '${tensorInfo.name}' requires FP32 dtype, got ${dtype.simpleName}"
                        }
                        when (fp16Policy) {
                            NarrowFloatLoadPolicy.DEQUANT_TO_FP32 -> {
                                val floats = dequantF16(bytes)
                                ctx.wrapFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
                            }
                            NarrowFloatLoadPolicy.KEEP_NATIVE -> {
                                // Mirrors the BF16 arm below: wrap the on-disk F16 bytes directly.
                                // dtype stays FP32 from the consumer's POV (the tensor data decodes
                                // on read); the storage type is what a narrow-float matmul dispatch
                                // pattern-matches on.
                                val fp16Data = Fp16DenseTensorData(shape, bytes)
                                ctx.fromData(fp16Data as TensorData<T, V>, dtype)
                            }
                        }
                    }

                    DataType.BFLOAT16 -> {
                        require(dtype == FP32::class) {
                            "SafeTensors BF16 tensor '${tensorInfo.name}' requires FP32 dtype, got ${dtype.simpleName}"
                        }
                        when (bf16Policy) {
                            Bf16LoadPolicy.DEQUANT_TO_FP32 -> {
                                val floats = dequantBF16(bytes)
                                ctx.wrapFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
                            }
                            Bf16LoadPolicy.KEEP_NATIVE -> {
                                // Wrap the on-disk BF16 bytes directly. dtype stays FP32 from
                                // the consumer's POV (Bf16TensorData : TensorData<DType, Float>
                                // decodes on read); the storage type is what the matmul
                                // dispatch will pattern-match on to pick the BF16 SPI kernel.
                                val bf16Data = Bf16DenseTensorData(shape, bytes)
                                ctx.fromData(bf16Data as TensorData<T, V>, dtype)
                            }
                        }
                    }

                    DataType.INT32 -> {
                        require(dtype == Int32::class) {
                            "SafeTensors I32 tensor '${tensorInfo.name}' requires Int32 dtype, got ${dtype.simpleName}"
                        }
                        val ints = bytesToIntArray(bytes)
                        ctx.wrapIntArray<T, Int>(shape, dtype, ints) as Tensor<T, V>
                    }

                    DataType.INT64 -> {
                        require(dtype == Int32::class) {
                            "SafeTensors I64 tensor '${tensorInfo.name}' requires Int32 dtype (downcast), got ${dtype.simpleName}"
                        }
                        println("WARNING: Downcasting I64 tensor '${tensorInfo.name}' to I32")
                        val longs = bytesToLongArray(bytes)
                        val ints = IntArray(longs.size) { longs[it].toInt() }
                        ctx.wrapIntArray<T, Int>(shape, dtype, ints) as Tensor<T, V>
                    }

                    DataType.INT8 -> {
                        require(dtype == Int8::class) {
                            "SafeTensors I8 tensor '${tensorInfo.name}' requires Int8 dtype, got ${dtype.simpleName}"
                        }
                        ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
                    }

                    DataType.UINT8 -> {
                        require(dtype == Int8::class) {
                            "SafeTensors U8 tensor '${tensorInfo.name}' requires Int8 dtype, got ${dtype.simpleName}"
                        }
                        // U8 stored as signed bytes (reinterpret)
                        ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
                    }

                    DataType.INT16, DataType.UINT16,
                    DataType.UINT32, DataType.UINT64 -> {
                        // Store as raw bytes for now
                        require(dtype == Int8::class) {
                            "SafeTensors ${tensorInfo.dtype} tensor '${tensorInfo.name}' requires Int8 dtype (raw bytes), got ${dtype.simpleName}"
                        }
                        ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
                    }

                    DataType.BOOL -> {
                        require(dtype == Int8::class) {
                            "SafeTensors BOOL tensor '${tensorInfo.name}' requires Int8 dtype, got ${dtype.simpleName}"
                        }
                        ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
                    }

                    DataType.UNKNOWN -> {
                        println("WARNING: Unknown dtype '${tensorInfo.dtype}' for tensor '${tensorInfo.name}'. Storing as raw bytes.")
                        require(dtype == Int8::class) {
                            "Unknown SafeTensors dtype requires Int8 dtype for raw bytes storage"
                        }
                        ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
                    }

                    else -> {
                        error("Unsupported SafeTensors dtype: ${tensorInfo.dataType} for tensor '${tensorInfo.name}'")
                    }
                }

                onTensorLoaded(tensorInfo.name, tensor)
                current++
                onProgress(current, total, tensorInfo.name)
            }
        }
    }

    // ========== Byte Conversion Helpers ==========

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            val offset = i * 4
            val bits = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
            out[i] = Float.fromBits(bits)
        }
        return out
    }

    private fun bytesToDoubleArray(bytes: ByteArray): DoubleArray {
        val out = DoubleArray(bytes.size / 8)
        for (i in out.indices) {
            val offset = i * 8
            val bits = (bytes[offset].toLong() and 0xFF) or
                    ((bytes[offset + 1].toLong() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toLong() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toLong() and 0xFF) shl 24) or
                    ((bytes[offset + 4].toLong() and 0xFF) shl 32) or
                    ((bytes[offset + 5].toLong() and 0xFF) shl 40) or
                    ((bytes[offset + 6].toLong() and 0xFF) shl 48) or
                    ((bytes[offset + 7].toLong() and 0xFF) shl 56)
            out[i] = Double.fromBits(bits)
        }
        return out
    }

    private fun bytesToIntArray(bytes: ByteArray): IntArray {
        val out = IntArray(bytes.size / 4)
        for (i in out.indices) {
            val offset = i * 4
            out[i] = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
        }
        return out
    }

    private fun bytesToLongArray(bytes: ByteArray): LongArray {
        val out = LongArray(bytes.size / 8)
        for (i in out.indices) {
            val offset = i * 8
            out[i] = (bytes[offset].toLong() and 0xFF) or
                    ((bytes[offset + 1].toLong() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toLong() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toLong() and 0xFF) shl 24) or
                    ((bytes[offset + 4].toLong() and 0xFF) shl 32) or
                    ((bytes[offset + 5].toLong() and 0xFF) shl 40) or
                    ((bytes[offset + 6].toLong() and 0xFF) shl 48) or
                    ((bytes[offset + 7].toLong() and 0xFF) shl 56)
        }
        return out
    }

    // ========== Dequantization Helpers ==========

    private fun dequantF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val half = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8)
            out[i] = halfToFloat(half)
        }
        return out
    }

    private fun dequantBF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val bf16Low = bytes[offset].toInt() and 0xFF
            val bf16High = bytes[offset + 1].toInt() and 0xFF
            // BF16 is just the upper 16 bits of F32
            val bits = (bf16High shl 24) or (bf16Low shl 16)
            out[i] = Float.fromBits(bits)
        }
        return out
    }

    private fun halfToFloat(hbits: Int): Float {
        val mant = hbits and 0x03FF
        val exp = hbits and 0x7C00
        val sign = hbits and 0x8000
        return when (exp) {
            0 -> {
                // Subnormal
                val v = (mant.toFloat() / 1024.0f) * (2.0f).pow(-14)
                if (sign != 0) -v else v
            }
            0x7C00 -> {
                // Inf/NaN
                val v = if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                if (sign != 0) -v else v
            }
            else -> {
                // Normal
                val v = (1.0f + mant.toFloat() / 1024.0f) * (2.0f).pow((exp shr 10) - 15)
                if (sign != 0) -v else v
            }
        }
    }

    companion object {

        /**
         * Constructs a SafeTensorsParametersLoader from a generalised
         * [DTypePolicy] instead of the BF16-specific [Bf16LoadPolicy].
         * Bridge for the policy-driven loader path described in the
         * dtype-policy RFC (#615).
         *
         * Policy → behaviour mapping (BF16 source tensors only —
         * other dtypes are handled per the per-arm `require` checks
         * in [load]):
         * - [DTypePolicy.Any]: BF16 dequants to FP32 (the existing
         *   adaptive default).
         * - [DTypePolicy.Require] target = `BF16`: KEEP_NATIVE.
         * - [DTypePolicy.Require] target = `FP32`: DEQUANT_TO_FP32.
         * - [DTypePolicy.Require] target = `FP16`: throws — F16
         *   KEEP_NATIVE is a follow-up (no `Fp16DenseTensorData`
         *   yet); use `Require(FP32)` if you want F16 dequanted, or
         *   `Any` to inherit the adaptive default.
         * - [DTypePolicy.Require] target = anything else: throws —
         *   SafeTensors can't fabricate dtypes the file doesn't carry.
         * - [DTypePolicy.Prefer] target = `BF16`: KEEP_NATIVE.
         * - [DTypePolicy.Prefer] target = anything else: DEQUANT_TO_FP32
         *   (the soft path falls through).
         * - [DTypePolicy.OneOf] containing `BF16`: KEEP_NATIVE.
         * - [DTypePolicy.OneOf] without `BF16`: DEQUANT_TO_FP32.
         */
        fun withPolicy(
            sourceProvider: () -> RandomAccessSource,
            policy: DTypePolicy,
            onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
        ): SafeTensorsParametersLoader = SafeTensorsParametersLoader(
            sourceProvider = sourceProvider,
            onProgress = onProgress,
            bf16Policy = mapPolicyToBf16(policy),
            fp16Policy = mapPolicyToFp16(policy),
        )

        internal fun mapPolicyToBf16(policy: DTypePolicy): Bf16LoadPolicy =
            mapPolicyToNarrow(policy, BF16)

        internal fun mapPolicyToFp16(policy: DTypePolicy): NarrowFloatLoadPolicy =
            mapPolicyToNarrow(policy, FP16)

        /**
         * Resolve [policy] for one narrow-float source format. A tensor is kept native only when
         * the policy names *that* format — `Require(BF16)` must not keep F16 tensors packed, and
         * vice versa, since neither can be converted to the other without a lossy re-encode.
         */
        private fun mapPolicyToNarrow(policy: DTypePolicy, native: DType): NarrowFloatLoadPolicy =
            when (policy) {
                DTypePolicy.Any -> NarrowFloatLoadPolicy.DEQUANT_TO_FP32
                is DTypePolicy.Require -> when (policy.target) {
                    native -> NarrowFloatLoadPolicy.KEEP_NATIVE
                    // The other narrow format, or FP32: this format still widens.
                    BF16, FP16, FP32 -> NarrowFloatLoadPolicy.DEQUANT_TO_FP32
                    else -> throw IllegalArgumentException(
                        "SafeTensorsParametersLoader: Require(${policy.target.name}) is not satisfiable — " +
                            "the loader produces FP32 / BF16 / FP16 / Int32 / Int8 tensors depending on " +
                            "source dtype; it cannot fabricate ${policy.target.name} from arbitrary sources.",
                    )
                }
                is DTypePolicy.Prefer -> if (policy.target == native) NarrowFloatLoadPolicy.KEEP_NATIVE
                                        else NarrowFloatLoadPolicy.DEQUANT_TO_FP32
                is DTypePolicy.OneOf -> if (native in policy.allowed) NarrowFloatLoadPolicy.KEEP_NATIVE
                                       else NarrowFloatLoadPolicy.DEQUANT_TO_FP32
            }
    }
}
