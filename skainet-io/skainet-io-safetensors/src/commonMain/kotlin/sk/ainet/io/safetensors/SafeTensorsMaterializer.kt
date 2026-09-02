package sk.ainet.io.safetensors

import sk.ainet.context.ExecutionContext
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
 * Shared per-tensor materialization for SafeTensors loaders.
 *
 * Owns the dtype-dispatch (raw little-endian bytes → typed [Tensor]) and the
 * narrow-float policy handling used by both [SafeTensorsParametersLoader]
 * (single file) and [ShardedSafeTensorsParametersLoader] (index + shards).
 *
 * The signature is deliberately primitive-typed (name/dataType/shape/bytes)
 * rather than taking a tensor-info object: the single-file reader surfaces
 * [StreamingSafeTensorInfo] while the sharded reader surfaces
 * [ShardedTensorInfo], and the two are unrelated types.
 */
internal object SafeTensorsMaterializer {

    /**
     * Materialize one tensor from its raw on-disk bytes.
     *
     * Conversion rules (identical to the historical
     * [SafeTensorsParametersLoader] behavior):
     * - F32/F64 → FP32 (F64 downcast with warning)
     * - F16 → FP32 dequant, or native [Fp16DenseTensorData] under KEEP_NATIVE
     * - BF16 → FP32 dequant, or native [Bf16DenseTensorData] under KEEP_NATIVE
     * - I32/I64 → Int32 (I64 downcast with warning)
     * - I8/U8/I16/U16/U32/U64/BOOL/UNKNOWN → Int8 raw bytes
     *
     * Each arm `require`s the matching requested [dtype] and throws otherwise.
     */
    @Suppress("UNCHECKED_CAST")
    fun <T : DType, V> materialize(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        name: String,
        dataType: DataType,
        rawDtype: String,
        shape: Shape,
        bytes: ByteArray,
        bf16Policy: Bf16LoadPolicy,
        fp16Policy: NarrowFloatLoadPolicy,
    ): Tensor<T, V> = when (dataType) {
        DataType.FLOAT32 -> {
            require(dtype == FP32::class) {
                "SafeTensors F32 tensor '$name' requires FP32 dtype, got ${dtype.simpleName}"
            }
            val floats = bytesToFloatArray(bytes)
            // Wrap the decoded array (zero-copy) — it was freshly allocated by bytesToFloatArray
            ctx.wrapFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
        }

        DataType.FLOAT64 -> {
            require(dtype == FP32::class) {
                "SafeTensors F64 tensor '$name' requires FP32 dtype (downcast), got ${dtype.simpleName}"
            }
            println("WARNING: Downcasting F64 tensor '$name' to F32")
            val doubles = bytesToDoubleArray(bytes)
            val floats = FloatArray(doubles.size) { doubles[it].toFloat() }
            ctx.wrapFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
        }

        DataType.FLOAT16 -> {
            require(dtype == FP32::class) {
                "SafeTensors F16 tensor '$name' requires FP32 dtype, got ${dtype.simpleName}"
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
                "SafeTensors BF16 tensor '$name' requires FP32 dtype, got ${dtype.simpleName}"
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
                "SafeTensors I32 tensor '$name' requires Int32 dtype, got ${dtype.simpleName}"
            }
            val ints = bytesToIntArray(bytes)
            ctx.wrapIntArray<T, Int>(shape, dtype, ints) as Tensor<T, V>
        }

        DataType.INT64 -> {
            require(dtype == Int32::class) {
                "SafeTensors I64 tensor '$name' requires Int32 dtype (downcast), got ${dtype.simpleName}"
            }
            println("WARNING: Downcasting I64 tensor '$name' to I32")
            val longs = bytesToLongArray(bytes)
            val ints = IntArray(longs.size) { longs[it].toInt() }
            ctx.wrapIntArray<T, Int>(shape, dtype, ints) as Tensor<T, V>
        }

        DataType.INT8 -> {
            require(dtype == Int8::class) {
                "SafeTensors I8 tensor '$name' requires Int8 dtype, got ${dtype.simpleName}"
            }
            ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
        }

        DataType.UINT8 -> {
            require(dtype == Int8::class) {
                "SafeTensors U8 tensor '$name' requires Int8 dtype, got ${dtype.simpleName}"
            }
            // U8 stored as signed bytes (reinterpret)
            ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
        }

        DataType.INT16, DataType.UINT16,
        DataType.UINT32, DataType.UINT64 -> {
            // Store as raw bytes for now
            require(dtype == Int8::class) {
                "SafeTensors $rawDtype tensor '$name' requires Int8 dtype (raw bytes), got ${dtype.simpleName}"
            }
            ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
        }

        DataType.BOOL -> {
            require(dtype == Int8::class) {
                "SafeTensors BOOL tensor '$name' requires Int8 dtype, got ${dtype.simpleName}"
            }
            ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
        }

        DataType.UNKNOWN -> {
            println("WARNING: Unknown dtype '$rawDtype' for tensor '$name'. Storing as raw bytes.")
            require(dtype == Int8::class) {
                "Unknown SafeTensors dtype requires Int8 dtype for raw bytes storage"
            }
            ctx.fromByteArray<T, Byte>(shape, dtype, bytes) as Tensor<T, V>
        }

        else -> {
            error("Unsupported SafeTensors dtype: $dataType for tensor '$name'")
        }
    }

    /**
     * The dtype the requested [dtype] KClass must be for a tensor of
     * [dataType] to materialize, or `null` when [materialize] accepts it
     * under any policy. Used by fail-fast pre-scans to reject a load
     * before any tensor is delivered.
     */
    fun requiredDType(dataType: DataType): KClass<out DType> = when (dataType) {
        DataType.FLOAT32, DataType.FLOAT64, DataType.FLOAT16, DataType.BFLOAT16 -> FP32::class
        DataType.INT32, DataType.INT64 -> Int32::class
        else -> Int8::class
    }

    // ========== Byte Conversion Helpers ==========

    internal fun bytesToFloatArray(bytes: ByteArray): FloatArray {
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

    internal fun bytesToDoubleArray(bytes: ByteArray): DoubleArray {
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

    internal fun bytesToIntArray(bytes: ByteArray): IntArray {
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

    internal fun bytesToLongArray(bytes: ByteArray): LongArray {
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

    internal fun dequantF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val half = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8)
            out[i] = halfToFloat(half)
        }
        return out
    }

    internal fun dequantBF16(bytes: ByteArray): FloatArray {
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

    // ========== Policy Mapping ==========

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
