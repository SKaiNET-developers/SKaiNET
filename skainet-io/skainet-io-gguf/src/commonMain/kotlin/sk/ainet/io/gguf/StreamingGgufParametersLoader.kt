package sk.ainet.io.gguf

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.types.DType
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
