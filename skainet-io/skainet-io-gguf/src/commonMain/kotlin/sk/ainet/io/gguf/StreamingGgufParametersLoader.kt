package sk.ainet.io.gguf

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.reflect.KClass

/**
 * Streaming GGUF parameters loader — the recommended path for loading GGUF models.
 *
 * Unlike [GgufParametersLoader] (which uses the legacy [GGUFReader] and rejects
 * quantized types), this loader:
 * - Uses [StreamingGGUFReader] for memory-efficient parsing
 * - Supports quantized types ([SUPPORTED_TENSOR_TYPES]) as packed [TensorData]
 * - Loads tensor data on-demand without heap-loading the full file
 * - Preserves quantized layout through the loading pipeline
 *
 * For F32 and I32 tensors, data is returned as standard dense arrays.
 * For quantized tensors, data is returned as packed block storage
 * (e.g., [Q4_KBlockTensorData], [Q8_0BlockTensorData]).
 *
 * A file containing tensors outside [SUPPORTED_TENSOR_TYPES] (e.g. Q4_1)
 * fails fast: [load] throws before any tensor is delivered, naming the
 * offending tensors and the supported set, instead of silently skipping
 * them and letting the missing weights crash the forward pass later
 * (#919).
 */
public class StreamingGgufParametersLoader(
    private val sourceProvider: () -> RandomAccessSource,
    private val onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
    /**
     * Keep `F16` source tensors in their on-disk 2-bytes-per-element layout instead of widening
     * them to FP32 at load. Off by default — flip via `withPolicy(Require(FP16))`.
     */
    private val keepF16Native: Boolean = false,
    /**
     * Keep `BF16` source tensors packed. Off by default — flip via `withPolicy(Require(BF16))`.
     */
    private val keepBf16Native: Boolean = false,
    /**
     * How quantized tensors are materialized (#782).
     *
     * - [QuantPolicy.NATIVE_OPTIMIZED] (default — the loader's historical behavior):
     *   quantized tensors are delivered as packed block [TensorData]; F32/F16/BF16
     *   are dense FP32 (subject to [keepF16Native]/[keepBf16Native]).
     * - [QuantPolicy.DEQUANTIZE_TO_FP32]: quantized tensors are dequantized
     *   *streaming, per tensor, block-by-block into the destination `FloatArray`*,
     *   which is then wrapped zero-copy. Peak transient memory per tensor is the
     *   packed source bytes only — there is no full-size intermediate copy.
     * - [QuantPolicy.RAW_BYTES] is not supported by this loader (it preserves
     *   packed block storage instead) and is rejected eagerly.
     */
    private val quantPolicy: QuantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
) : ParametersLoader {

    init {
        require(quantPolicy != QuantPolicy.RAW_BYTES) {
            "StreamingGgufParametersLoader does not support QuantPolicy.RAW_BYTES — quantized " +
                "tensors are preserved as packed block TensorData (NATIVE_OPTIMIZED) or " +
                "dequantized to dense FP32 (DEQUANTIZE_TO_FP32)."
        }
    }

    @Suppress("UNCHECKED_CAST")
    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        StreamingGGUFReader.open(sourceProvider()).use { reader ->
            val tensors = reader.tensors
            failFastOnUnsupportedTensorTypes(tensors)
            val total = tensors.size.toLong()
            var current = 0L

            for (tensorInfo in tensors) {
                val shape = Shape(*tensorInfo.shape.map { it.toInt() }.toIntArray())
                val rawBytes = reader.loadTensorData(tensorInfo)

                val tensor: Tensor<T, V>? = when (tensorInfo.tensorType) {
                    GGMLQuantizationType.F32 -> {
                        when (dtype) {
                            // The freshly decoded array is loader-owned — wrap it zero-copy
                            // instead of paying the factory's defensive copy (#782).
                            FP32::class -> ctx.wrapFloatArray<T, Float>(shape, dtype, bytesToFloatArray(rawBytes)) as Tensor<T, V>
                            else -> null
                        }
                    }

                    GGMLQuantizationType.I32 -> {
                        val ints = bytesToIntArray(rawBytes)
                        when (dtype) {
                            Int32::class -> ctx.fromIntArray<T, Int>(shape, dtype, ints) as Tensor<T, V>
                            else -> null
                        }
                    }

                    GGMLQuantizationType.F16 -> when (dtype) {
                        FP32::class -> if (keepF16Native) {
                            // Zero-widening path: hand the on-disk bytes straight through as
                            // packed binary16. Consumers still see Float on read.
                            @Suppress("UNCHECKED_CAST")
                            val packed = Fp16DenseTensorData(shape, rawBytes)
                            ctx.fromData<T, V>(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                        } else {
                            // Loader-owned widened array — zero-copy wrap (#782).
                            ctx.wrapFloatArray<T, Float>(shape, dtype, dequantF16(rawBytes)) as Tensor<T, V>
                        }
                        else -> null
                    }

                    GGMLQuantizationType.BF16 -> when (dtype) {
                        FP32::class -> if (keepBf16Native) {
                            @Suppress("UNCHECKED_CAST")
                            val packed = Bf16DenseTensorData(shape, rawBytes)
                            ctx.fromData<T, V>(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                        } else {
                            // Loader-owned widened array — zero-copy wrap (#782).
                            ctx.wrapFloatArray<T, Float>(shape, dtype, dequantBF16(rawBytes)) as Tensor<T, V>
                        }
                        else -> null
                    }

                    GGMLQuantizationType.Q4_K,
                    GGMLQuantizationType.Q5_K,
                    GGMLQuantizationType.Q6_K,
                    GGMLQuantizationType.Q8_0,
                    GGMLQuantizationType.Q4_0,
                    GGMLQuantizationType.Q5_0,
                    GGMLQuantizationType.Q5_1 -> quantizedTensor(ctx, dtype, shape, tensorInfo, rawBytes)

                    else -> throw IllegalStateException(
                        "StreamingGgufParametersLoader: tensor '${tensorInfo.name}' of type " +
                            "${tensorInfo.tensorType} passed the load-time pre-scan but has no load " +
                            "branch — SUPPORTED_TENSOR_TYPES and this when-expression have drifted. " +
                            "Please report this as a bug."
                    )
                }

                if (tensor != null) {
                    onTensorLoaded(tensorInfo.name, tensor)
                }

                current += 1
                onProgress(current, total, tensorInfo.name)
            }
        }
    }

    /**
     * Materialize a quantized tensor according to [quantPolicy].
     *
     * DEQUANTIZE_TO_FP32 (#782): the packed bytes are unpacked block-by-block
     * straight into one destination `FloatArray` (the shared [DequantOps]
     * kernels write each block into the single output array — no boxed values,
     * no per-tensor intermediate), and the destination is wrapped zero-copy.
     * Peak transient allocation per tensor is the packed source bytes.
     *
     * Any other policy (or a non-float destination dtype) preserves the packed
     * block storage exactly as before.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> quantizedTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        shape: Shape,
        tensorInfo: StreamingTensorInfo,
        rawBytes: ByteArray,
    ): Tensor<T, V> {
        if (quantPolicy == QuantPolicy.DEQUANTIZE_TO_FP32 &&
            (dtype == FP32::class || dtype == FP16::class)
        ) {
            val dest = DequantOps.dequantFromBytes(rawBytes, tensorInfo.tensorType, tensorInfo.nElements.toInt())
            return ctx.wrapFloatArray<T, Float>(shape, dtype, dest) as Tensor<T, V>
        }
        val packed = when (tensorInfo.tensorType) {
            GGMLQuantizationType.Q4_K -> Q4_KBlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q5_K -> Q5_KBlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q6_K -> Q6_KBlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q8_0 -> Q8_0BlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q4_0 -> Q4_0BlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q5_0 -> Q5_0BlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q5_1 -> Q5_1BlockTensorData.fromRawBytes(shape, rawBytes)
            else -> throw IllegalStateException(
                "quantizedTensor called for non-quantized type ${tensorInfo.tensorType}"
            )
        }
        return ctx.fromData(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val count = bytes.size / 4
        return FloatArray(count) { i ->
            val off = i * 4
            Float.fromBits(
                (bytes[off].toInt() and 0xFF) or
                    ((bytes[off + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[off + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[off + 3].toInt() and 0xFF) shl 24)
            )
        }
    }

    private fun bytesToIntArray(bytes: ByteArray): IntArray {
        val count = bytes.size / 4
        return IntArray(count) { i ->
            val off = i * 4
            (bytes[off].toInt() and 0xFF) or
                ((bytes[off + 1].toInt() and 0xFF) shl 8) or
                ((bytes[off + 2].toInt() and 0xFF) shl 16) or
                ((bytes[off + 3].toInt() and 0xFF) shl 24)
        }
    }

    private fun dequantF16(bytes: ByteArray): FloatArray {
        val count = bytes.size / 2
        return FloatArray(count) { i ->
            val off = i * 2
            val halfBits = (bytes[off].toInt() and 0xFF) or
                ((bytes[off + 1].toInt() and 0xFF) shl 8)
            halfToFloat(halfBits)
        }
    }

    private fun dequantBF16(bytes: ByteArray): FloatArray {
        val count = bytes.size / 2
        return FloatArray(count) { i ->
            val off = i * 2
            val bf16Bits = (bytes[off].toInt() and 0xFF) or
                ((bytes[off + 1].toInt() and 0xFF) shl 8)
            Float.fromBits(bf16Bits shl 16)
        }
    }

    public companion object {

        /**
         * The tensor types [load] can materialize. The when-expression in [load]
         * and the eager pre-scan both derive from this set, so a type added to
         * one place cannot silently drift from the other.
         */
        public val SUPPORTED_TENSOR_TYPES: Set<GGMLQuantizationType> = setOf(
            GGMLQuantizationType.F32,
            GGMLQuantizationType.I32,
            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16,
            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_0,
        )

        private const val MAX_LISTED_TENSORS = 8

        /**
         * Eager pre-scan over the file's tensor directory: throws before any
         * tensor is delivered if the file contains types this loader cannot
         * materialize. This follows the RFC's "fail before execution" rule
         * (see [withPolicy]) — the alternative, skipping the tensor, produces
         * a model with silently missing weights whose failure surfaces far
         * away in the forward pass (#919).
         */
        internal fun failFastOnUnsupportedTensorTypes(tensors: List<StreamingTensorInfo>) {
            val unsupported = tensors.filter { it.tensorType !in SUPPORTED_TENSOR_TYPES }
            if (unsupported.isEmpty()) return

            val listed = unsupported.take(MAX_LISTED_TENSORS).joinToString(", ") {
                val type = if (it.isUnknownType) "unknown type value ${it.rawTypeValue}" else it.tensorType.name
                "'${it.name}' ($type)"
            }
            val more = if (unsupported.size > MAX_LISTED_TENSORS) {
                " and ${unsupported.size - MAX_LISTED_TENSORS} more"
            } else {
                ""
            }
            throw IllegalArgumentException(
                "GGUF contains ${unsupported.size} tensor(s) with quantization types this loader " +
                    "does not support: $listed$more. Supported types: " +
                    "${SUPPORTED_TENSOR_TYPES.joinToString(", ") { it.name }}. " +
                    "Re-quantize the model to a supported format (e.g. Q8_0, Q4_0, Q4_K or F16).",
            )
        }

        /**
         * Convenience constructor that takes a [DTypePolicy] and
         * validates it against the dtypes the GGUF loader supports
         * today. The validator runs eagerly — if the requested
         * policy can never be satisfied by this loader (e.g.
         * `Require(Int8)` against a GGUF file: this loader doesn't
         * cast), an [IllegalArgumentException] is raised before the
         * loader is constructed, exactly matching the RFC's
         * "fail before execution" rule.
         *
         * Current per-source behaviour the validator enforces:
         * - GGUF `F32` / `I32` / `Q4_K` / `Q8_0` are always
         *   preserved verbatim — any policy that admits the
         *   matching dtype passes.
         * - GGUF `F16` / `BF16` always dequant to FP32 in this
         *   loader today (no KEEP_NATIVE GGUF path yet). A policy
         *   of `Require(BF16)` or `Require(FP16)` therefore fails
         *   eagerly; use `Any`, `Prefer`, or `OneOf` containing
         *   `FP32` if you want the adaptive dequant behaviour.
         *
         * The validator is conservative — it doesn't open the GGUF
         * file to check which dtypes are actually present. A
         * policy that's satisfiable in principle but happens to
         * conflict with the specific file's tensors will surface at
         * iteration time via the `null`-return path in [load].
         * Tensor *types* outside [SUPPORTED_TENSOR_TYPES], by
         * contrast, fail eagerly once the file is opened — see
         * [failFastOnUnsupportedTensorTypes].
         */
        public fun withPolicy(
            sourceProvider: () -> RandomAccessSource,
            policy: DTypePolicy,
            onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
        ): StreamingGgufParametersLoader {
            validatePolicy(policy)
            return StreamingGgufParametersLoader(
                sourceProvider = sourceProvider,
                onProgress = onProgress,
                keepF16Native = keepsNative(policy, FP16),
                keepBf16Native = keepsNative(policy, BF16),
            )
        }

        /**
         * Whether [policy] asks for [native] tensors to stay in their on-disk 16-bit layout.
         *
         * Only the format the policy actually names is kept — neither narrow format can be turned
         * into the other without a lossy re-encode, so `Require(BF16)` must still widen F16 sources.
         */
        internal fun keepsNative(policy: DTypePolicy, native: DType): Boolean = when (policy) {
            DTypePolicy.Any -> false
            is DTypePolicy.Require -> policy.target == native
            is DTypePolicy.Prefer -> policy.target == native
            is DTypePolicy.OneOf -> native in policy.allowed
        }

        internal fun validatePolicy(policy: DTypePolicy) {
            when (policy) {
                DTypePolicy.Any -> Unit
                is DTypePolicy.Prefer -> Unit
                is DTypePolicy.OneOf -> Unit
                is DTypePolicy.Require -> when (policy.target) {
                    // FP16 / BF16 are satisfiable for sources already in that format: the loader
                    // hands the packed bytes through via Fp16/Bf16DenseTensorData. Sources in any
                    // other format still widen to FP32 rather than being re-encoded.
                    FP32, FP16, BF16 -> Unit
                    else -> throw IllegalArgumentException(
                        "StreamingGgufParametersLoader: Require(${policy.target.name}) is not satisfiable — " +
                            "this loader preserves source tensors (dense FP32/Int32 or packed quantized " +
                            "blocks) and does not cast between dtypes. Use Any to inherit the source dtype, " +
                            "or open a follow-up to add a ${policy.target.name} cast path.",
                    )
                }
            }
        }
    }

    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits and 0x8000) shl 16
        val exp = (hbits and 0x7C00) shr 10
        val mant = hbits and 0x03FF

        return when (exp) {
            0 -> {
                if (mant == 0) Float.fromBits(sign)
                else {
                    var m = mant; var e = -14
                    while ((m and 0x400) == 0) { m = m shl 1; e-- }
                    m = m and 0x3FF
                    Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
                }
            }
            31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }
}
