package sk.ainet.io.gguf

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
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
 * - Supports quantized types (Q4_K, Q8_0) as packed [TensorData]
 * - Loads tensor data on-demand without heap-loading the full file
 * - Preserves quantized layout through the loading pipeline
 *
 * For F32 and I32 tensors, data is returned as standard dense arrays.
 * For quantized tensors, data is returned as packed block storage
 * (e.g., [Q4_KBlockTensorData], [Q8_0BlockTensorData]).
 */
public class StreamingGgufParametersLoader(
    private val sourceProvider: () -> RandomAccessSource,
    private val onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> }
) : ParametersLoader {

    @Suppress("UNCHECKED_CAST")
    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        StreamingGGUFReader.open(sourceProvider()).use { reader ->
            val tensors = reader.tensors
            val total = tensors.size.toLong()
            var current = 0L

            for (tensorInfo in tensors) {
                val shape = Shape(*tensorInfo.shape.map { it.toInt() }.toIntArray())
                val rawBytes = reader.loadTensorData(tensorInfo)

                val tensor: Tensor<T, V>? = when (tensorInfo.tensorType) {
                    GGMLQuantizationType.F32 -> {
                        val floats = bytesToFloatArray(rawBytes)
                        when (dtype) {
                            FP32::class -> ctx.fromFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
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

                    GGMLQuantizationType.F16 -> {
                        val floats = dequantF16(rawBytes)
                        when (dtype) {
                            FP32::class -> ctx.fromFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
                            else -> null
                        }
                    }

                    GGMLQuantizationType.BF16 -> {
                        val floats = dequantBF16(rawBytes)
                        when (dtype) {
                            FP32::class -> ctx.fromFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
                            else -> null
                        }
                    }

                    GGMLQuantizationType.Q4_K -> {
                        @Suppress("UNCHECKED_CAST")
                        val packed = Q4_KBlockTensorData.fromRawBytes(shape, rawBytes)
                        ctx.fromData<T, V>(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                    }

                    GGMLQuantizationType.Q5_K -> {
                        @Suppress("UNCHECKED_CAST")
                        val packed = Q5_KBlockTensorData.fromRawBytes(shape, rawBytes)
                        ctx.fromData<T, V>(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                    }

                    GGMLQuantizationType.Q6_K -> {
                        @Suppress("UNCHECKED_CAST")
                        val packed = Q6_KBlockTensorData.fromRawBytes(shape, rawBytes)
                        ctx.fromData<T, V>(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                    }

                    GGMLQuantizationType.Q8_0 -> {
                        @Suppress("UNCHECKED_CAST")
                        val packed = Q8_0BlockTensorData.fromRawBytes(shape, rawBytes)
                        ctx.fromData<T, V>(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                    }

                    else -> {
                        onProgress(current, total, "SKIP: ${tensorInfo.name} (unsupported type ${tensorInfo.tensorType})")
                        null
                    }
                }

                if (tensor != null) {
                    onTensorLoaded(tensorInfo.name, tensor)
                }

                current += 1
                onProgress(current, total, tensorInfo.name)
            }
        }
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
         */
        public fun withPolicy(
            sourceProvider: () -> RandomAccessSource,
            policy: DTypePolicy,
            onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
        ): StreamingGgufParametersLoader {
            validatePolicy(policy)
            return StreamingGgufParametersLoader(sourceProvider, onProgress)
        }

        internal fun validatePolicy(policy: DTypePolicy) {
            when (policy) {
                DTypePolicy.Any -> Unit
                is DTypePolicy.Prefer -> Unit
                is DTypePolicy.OneOf -> Unit
                is DTypePolicy.Require -> when (policy.target) {
                    FP32 -> Unit
                    BF16 -> throw IllegalArgumentException(
                        "StreamingGgufParametersLoader: Require(BF16) is not supported — " +
                            "GGUF BF16 sources are dequanted to FP32 by this loader today (no KEEP_NATIVE " +
                            "GGUF path yet). Use Any or Prefer(BF16) to accept the dequant fallback, or " +
                            "wait for the policy-aware GGUF reader to land.",
                    )
                    FP16 -> throw IllegalArgumentException(
                        "StreamingGgufParametersLoader: Require(FP16) is not supported — " +
                            "GGUF F16 sources are dequanted to FP32 by this loader today (no Fp16DenseTensorData " +
                            "backing yet). Use Any or Prefer(FP16) to accept the dequant fallback.",
                    )
                    else -> throw IllegalArgumentException(
                        "StreamingGgufParametersLoader: Require(${policy.target.name}) is not satisfiable — " +
                            "this loader produces FP32 / Int32 / Q4_K / Q8_0 tensors only, and does not cast " +
                            "between source dtypes. Use Any to inherit the source dtype, or open a follow-up " +
                            "to add a ${policy.target.name} cast path.",
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
