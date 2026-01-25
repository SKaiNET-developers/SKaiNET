package sk.ainet.io.onnx

import onnx.ModelProto
import onnx.TensorProto
import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightLoadResult
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.math.max

/**
 * Utility for loading ONNX model initializers (weights) into SKaiNET Module parameters.
 *
 * Features:
 * - ONNX-specific tensor decoding (FLOAT, INT32, INT64)
 * - Format-agnostic weight mapping via [WeightMapper]
 * - Support for multiple tensor data types (all converted to Float)
 * - Debug mode for troubleshooting weight mapping issues
 *
 * Example usage:
 * ```kotlin
 * val model = ModelProto.decodeFromByteArray(bytes)
 * val ctx = DirectCpuExecutionContext(phase = Phase.INFERENCE)
 * val loadResult = OnnxWeightLoader.loadInitializers(model, ctx)
 * val mapping = OnnxWeightLoader.applyWeights(module, loadResult.tensors)
 * ```
 */
public object OnnxWeightLoader {

    /**
     * Holds a decoded ONNX tensor ready for weight assignment.
     * @deprecated Use [WeightTensor] from skainet-io-core instead for format-agnostic code.
     */
    @Deprecated(
        message = "Use WeightTensor from skainet-io-core for format-agnostic code",
        replaceWith = ReplaceWith("WeightTensor<FP32, Float>", "sk.ainet.io.weights.WeightTensor")
    )
    public data class InitTensor(
        val name: String,
        val isBias: Boolean,
        val shape: List<Int>,
        val tensor: Tensor<FP32, Float>
    ) {
        /** Convert to format-agnostic WeightTensor. */
        public fun toWeightTensor(): WeightTensor<FP32, Float> = WeightTensor(
            name = name,
            shape = shape,
            tensor = tensor,
            isBias = isBias
        )
    }

    /**
     * Result of loading initializers from an ONNX model.
     * @deprecated Use [WeightLoadResult] from skainet-io-core instead.
     */
    @Deprecated(
        message = "Use WeightLoadResult from skainet-io-core",
        replaceWith = ReplaceWith("WeightLoadResult<FP32, Float>", "sk.ainet.io.weights.WeightLoadResult")
    )
    public data class InitializerLoadResult(
        val tensors: List<InitTensor>,
        val skipped: List<String>
    ) {
        /** Convert to format-agnostic WeightLoadResult. */
        public fun toWeightLoadResult(): WeightLoadResult<FP32, Float> = WeightLoadResult(
            tensors = tensors.map { it.toWeightTensor() },
            skipped = skipped
        )
    }


    /**
     * Load all initializers (weights) from an ONNX ModelProto.
     *
     * Skips running_mean and running_var tensors (batch norm statistics).
     *
     * @param model The ONNX model proto
     * @param ctx Execution context for tensor creation
     * @return Load result with decoded tensors and list of skipped tensor names
     */
    public fun loadInitializers(model: ModelProto, ctx: ExecutionContext): WeightLoadResult<FP32, Float> {
        val graph = requireNotNull(model.graph) { "Model does not contain a graph" }
        val skipped = mutableListOf<String>()
        val tensors = graph.initializer
            .filterNot {
                it.name.contains("running_mean", ignoreCase = true) ||
                    it.name.contains("running_var", ignoreCase = true)
            }
            .mapNotNull { tensor ->
                decodeInitializer(tensor, ctx)
                    .onFailure { skipped += "${tensor.name}: ${it.message ?: "unknown error"}" }
                    .getOrNull()
            }
        return WeightLoadResult(tensors, skipped)
    }

    /**
     * Load initializers returning the legacy InitializerLoadResult type.
     * @deprecated Use [loadInitializers] which returns [WeightLoadResult] instead.
     */
    @Deprecated(
        message = "Use loadInitializers which returns WeightLoadResult",
        replaceWith = ReplaceWith("loadInitializers(model, ctx)")
    )
    @Suppress("DEPRECATION")
    public fun loadInitializersLegacy(model: ModelProto, ctx: ExecutionContext): InitializerLoadResult {
        val result = loadInitializers(model, ctx)
        return InitializerLoadResult(
            tensors = result.tensors.map { InitTensor(it.name, it.isBias, it.shape, it.tensor) },
            skipped = result.skipped
        )
    }

    /**
     * Decode a single ONNX TensorProto into a WeightTensor.
     *
     * Supports FLOAT, INT32, and INT64 data types (all converted to Float).
     */
    public fun decodeInitializer(tensor: TensorProto, ctx: ExecutionContext): Result<WeightTensor<FP32, Float>> = runCatching {
        val t = tensor.toFloatTensorMultiDtype(ctx) ?: error("no data for dtype=${tensor.dataType}")
        WeightTensor(
            name = tensor.name,
            shape = tensor.dims.map { it.toInt() },
            tensor = t,
            isBias = tensor.name.lowercase().contains("bias")
        )
    }

    /**
     * Apply ONNX weights to a module using name-based matching with shape-based fallback.
     *
     * Delegates to [WeightMapper.applyWeights] for the actual mapping logic.
     *
     * @param module The SKaiNET module to apply weights to
     * @param tensors The decoded weight tensors
     * @return Mapping result with statistics
     */
    public fun <T : DType, V> applyWeights(
        module: Module<T, V>,
        tensors: List<WeightTensor<T, V>>
    ): WeightMapper.MappingResult = WeightMapper.applyWeights(module, tensors)

    /**
     * Apply weights with optional debug output.
     *
     * @param module The SKaiNET module to apply weights to
     * @param tensors The decoded weight tensors
     * @param debug Whether to print debug information
     * @return Mapping result with statistics
     */
    public fun <T : DType, V> applyWeightsWithDebug(
        module: Module<T, V>,
        tensors: List<WeightTensor<T, V>>,
        debug: Boolean
    ): WeightMapper.MappingResult = WeightMapper.applyWeights(
        module = module,
        tensors = tensors,
        config = MappingConfig(debug = debug)
    )

    /**
     * Apply weights from legacy InitTensor list.
     * @deprecated Use [applyWeights] with [WeightTensor] list instead.
     */
    @Deprecated(
        message = "Use applyWeights with WeightTensor list",
        replaceWith = ReplaceWith("applyWeights(module, tensors.map { it.toWeightTensor() })")
    )
    @Suppress("DEPRECATION")
    public fun <T : DType, V> applyWeightsFromInitTensors(
        module: Module<T, V>,
        tensors: List<InitTensor>
    ): WeightMapper.MappingResult {
        @Suppress("UNCHECKED_CAST")
        return applyWeights(module, tensors.map { it.toWeightTensor() } as List<WeightTensor<T, V>>)
    }

    /**
     * Validate that all module parameters were mapped from ONNX initializers.
     * Throws IllegalArgumentException if mapping is incomplete.
     *
     * Delegates to [WeightMapper.validateAllParametersMapped].
     */
    public fun validateAllParametersMapped(mapping: WeightMapper.MappingResult, skipped: List<String> = emptyList()) {
        WeightMapper.validateAllParametersMapped(mapping, skipped)
    }

    /**
     * Extract the most specific "model.X.Y.Z" pattern from a DSL module path.
     * E.g., "Yolo8/model.22.cv2.0/model.22.cv2.0.0/Conv2d-0" -> "model.22.cv2.0.0"
     *
     * Delegates to [WeightMapper.extractLayerNameFromPath].
     */
    public fun extractOnnxNameFromPath(path: String): String? =
        WeightMapper.extractLayerNameFromPath(path)

    /**
     * Normalize a DSL layer name to ONNX naming format.
     * DSL uses underscores in some places while ONNX uses dots.
     *
     * Delegates to [WeightMapper.normalizeToOnnxFormat].
     */
    public fun normalizeToOnnxFormat(name: String): String =
        WeightMapper.normalizeToOnnxFormat(name)

    /**
     * Check if an ONNX tensor name matches a DSL layer name.
     *
     * Delegates to [WeightMapper.matchesLayerName].
     */
    public fun matchesOnnxLayer(onnxName: String, dslLayerName: String?): Boolean =
        WeightMapper.matchesLayerName(onnxName, dslLayerName)

    // ========== ONNX-Specific Helpers ==========

    /**
     * Convert TensorProto to Float tensor, supporting multiple data types.
     */
    private fun TensorProto.toFloatTensorMultiDtype(ctx: ExecutionContext): Tensor<FP32, Float>? {
        val shape = Shape(*dims.map { it.toInt() }.toIntArray())
        val volume = if (shape.rank == 0) 1 else max(1, shape.dimensions.fold(1) { acc, d -> acc * d })

        val floats: FloatArray = when (TensorProto.DataType.fromValue(dataType)) {
            TensorProto.DataType.FLOAT -> when {
                floatData.isNotEmpty() -> floatData.toFloatArray()
                rawData.array.isNotEmpty() -> rawData.array.toFloatArrayLE()
                else -> FloatArray(volume) { 0f }
            }
            TensorProto.DataType.INT64 -> when {
                int64Data.isNotEmpty() -> int64Data.map { it.toFloat() }.toFloatArray()
                rawData.array.isNotEmpty() -> rawData.array.toLongArrayLE().map { it.toFloat() }.toFloatArray()
                else -> FloatArray(volume) { 0f }
            }
            TensorProto.DataType.INT32 -> when {
                int32Data.isNotEmpty() -> int32Data.map { it.toFloat() }.toFloatArray()
                rawData.array.isNotEmpty() -> rawData.array.toIntArrayLE().map { it.toFloat() }.toFloatArray()
                else -> FloatArray(volume) { 0f }
            }
            else -> return null
        }

        // Pad to expected size if needed
        val padded = if (floats.size >= volume) floats else {
            FloatArray(volume) { i -> floats.getOrElse(i) { 0f } }
        }

        return ctx.fromFloatArray(shape, FP32::class, padded)
    }

    private fun ByteArray.toFloatArrayLE(): FloatArray {
        val out = FloatArray(size / 4)
        var o = 0
        var i = 0
        while (i + 3 < size) {
            val bits = (this[i].toInt() and 0xFF) or
                ((this[i + 1].toInt() and 0xFF) shl 8) or
                ((this[i + 2].toInt() and 0xFF) shl 16) or
                ((this[i + 3].toInt() and 0xFF) shl 24)
            out[o++] = Float.fromBits(bits)
            i += 4
        }
        return out
    }

    private fun ByteArray.toLongArrayLE(): LongArray {
        val out = LongArray(size / 8)
        var o = 0
        var i = 0
        while (i + 7 < size) {
            val bits = (this[i].toLong() and 0xFF) or
                ((this[i + 1].toLong() and 0xFF) shl 8) or
                ((this[i + 2].toLong() and 0xFF) shl 16) or
                ((this[i + 3].toLong() and 0xFF) shl 24) or
                ((this[i + 4].toLong() and 0xFF) shl 32) or
                ((this[i + 5].toLong() and 0xFF) shl 40) or
                ((this[i + 6].toLong() and 0xFF) shl 48) or
                ((this[i + 7].toLong() and 0xFF) shl 56)
            out[o++] = bits
            i += 8
        }
        return out
    }

    private fun ByteArray.toIntArrayLE(): IntArray {
        val out = IntArray(size / 4)
        var o = 0
        var i = 0
        while (i + 3 < size) {
            out[o++] = (this[i].toInt() and 0xFF) or
                ((this[i + 1].toInt() and 0xFF) shl 8) or
                ((this[i + 2].toInt() and 0xFF) shl 16) or
                ((this[i + 3].toInt() and 0xFF) shl 24)
            i += 4
        }
        return out
    }
}
