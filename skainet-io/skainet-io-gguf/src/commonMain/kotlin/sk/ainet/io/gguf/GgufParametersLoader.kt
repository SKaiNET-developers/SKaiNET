package sk.ainet.io.gguf

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.reflect.KClass

/**
 * ParametersLoader implementation backed by GGUFReader.
 *
 * Notes:
 * - Currently supports loading tensors as FP32 or Int32. Other dtypes can be added as needed.
 * - For quantized GGML tensor payloads, this implementation does not perform dequantization and will throw.
 * - A lightweight progress callback can be provided to observe per-tensor progress (current/total/name).
 */
class GgufParametersLoader(
    private val sourceProvider: () -> Source,
    private val onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> }
) : ParametersLoader {

    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        sourceProvider().use { src ->
            val reader = GGUFReader(src, loadTensorData = true)
            val tensors = reader.tensors
            val total = tensors.size.toLong()
            var current = 0L

            for (rt in tensors) {
                // Only native numeric types supported currently; quantized not yet handled here
                when (rt.tensorType) {
                    GGMLQuantizationType.F32 -> {
                        val data = reader.materialize(rt) as List<Float>
                        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())
                        @Suppress("UNCHECKED_CAST")
                        val tensor: Tensor<T, V> = when (dtype) {
                            FP32::class -> ctx.fromFloatArray<T, Float>(shape, dtype, data.toFloatArray()) as Tensor<T, V>
                            else -> error("GGUF loader: requested dtype ${'$'}{dtype.simpleName} incompatible with F32 payload")
                        }
                        onTensorLoaded(rt.name, tensor)
                    }
                    GGMLQuantizationType.I32 -> {
                        val data = (reader.materialize(rt) as List<Int>).toIntArray()
                        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())
                        @Suppress("UNCHECKED_CAST")
                        val tensor: Tensor<T, V> = when (dtype) {
                            Int32::class -> ctx.fromIntArray<T, Int>(shape, dtype, data) as Tensor<T, V>
                            else -> error("GGUF loader: requested dtype ${'$'}{dtype.simpleName} incompatible with I32 payload")
                        }
                        onTensorLoaded(rt.name, tensor)
                    }
                    else -> error("GGUF loader: unsupported or quantized tensor type ${'$'}{rt.tensorType}")
                }
                current += 1
                onProgress(current, total, rt.name)
            }
        }
    }
}
