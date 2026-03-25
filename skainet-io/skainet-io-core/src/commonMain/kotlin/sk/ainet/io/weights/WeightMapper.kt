package sk.ainet.io.weights

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.types.DType

/**
 * Configuration for weight mapping behavior.
 *
 * @property usePathBasedMatching Whether to use DSL module path for name matching
 * @property fallbackToShapeMatching Whether to fall back to shape-only matching if name matching fails
 * @property debug Whether to print debug information during mapping
 * @property nameResolver Optional resolver that translates module paths/param names to tensor names.
 *                        When provided, this takes priority over the default ONNX-style path matching.
 */
public data class MappingConfig(
    val usePathBasedMatching: Boolean = true,
    val fallbackToShapeMatching: Boolean = true,
    val debug: Boolean = false,
    val nameResolver: WeightNameResolver? = null
)

/**
 * Unified weight mapping utility for applying loaded weights to SKaiNET modules.
 *
 * This object provides format-agnostic weight mapping logic that can be used
 * by any model loader (ONNX, GGUF, SafeTensors, etc.). It supports:
 *
 * - Path-based name matching with automatic normalization
 * - Shape-based fallback matching when names don't match
 * - Debug mode for troubleshooting mapping issues
 *
 * Example usage:
 * ```kotlin
 * val tensors: List<WeightTensor<FP32, Float>> = loader.loadWeights(modelFile)
 * val result = WeightMapper.applyWeights(module, tensors)
 * WeightMapper.validateAllParametersMapped(result)
 * ```
 */
public object WeightMapper {

    /**
     * Result of mapping weights to module parameters.
     *
     * @property mapped Number of parameters that were successfully mapped
     * @property total Total number of parameters in the module
     * @property missingParams List of parameters that could not be mapped (with shapes)
     * @property unusedTensors List of tensors that were not used (with shapes)
     */
    public data class MappingResult(
        val mapped: Int,
        val total: Int,
        val missingParams: List<String>,
        val unusedTensors: List<String>
    )

    /**
     * Internal holder for parameter with its module path.
     */
    private data class ParamWithPath(
        val param: ModuleParameter<*, *>,
        val modulePath: String
    )

    /**
     * Apply weights to a module using name-based matching with shape-based fallback.
     *
     * The matching algorithm:
     * 1. Extract the layer name from the DSL module path
     * 2. Normalize DSL naming to standard format (e.g., "_m0_cv1" -> ".m.0.cv1")
     * 3. Try to match by name first, then fall back to shape-only matching
     *
     * @param module The SKaiNET module to apply weights to
     * @param tensors The loaded weight tensors
     * @param config Configuration for mapping behavior
     * @return Mapping result with statistics
     */
    public fun <T : DType, V> applyWeights(
        module: Module<T, V>,
        tensors: List<WeightTensor<T, V>>,
        config: MappingConfig = MappingConfig()
    ): MappingResult {
        val paramsWithPath = collectParamsWithPath(module)

        // Filter out constant tensors (those with "/" in name or empty dims)
        val modelTensors = tensors.filter {
            !it.name.startsWith("/") && it.shape.isNotEmpty() && it.shape.all { d -> d > 0 }
        }

        if (config.debug) {
            println("\n=== Weight Tensors ===")
            modelTensors.take(30).forEach { println("  ${it.name}: ${it.shape}") }
            println("\n=== DSL Parameters with Paths ===")
            paramsWithPath.take(30).forEach {
                println("  ${it.modulePath} -> ${it.param.name}: ${it.param.value.shape.dimensions.toList()}")
            }
        }

        val used = mutableSetOf<String>()
        var mapped = 0
        val missing = mutableListOf<String>()
        val mappingLog = mutableListOf<String>()

        paramsWithPath.forEach { pwp ->
            val param = pwp.param
            val isBiasParam = param is ModuleParameter.BiasParameter
            val pShape = param.value.shape.dimensions.toList()

            // Strategy 1: Try the explicit name resolver (for LLM and custom mappings)
            val byResolver = if (config.nameResolver != null) {
                val resolvedName = config.nameResolver.resolve(pwp.modulePath, param.name)
                if (resolvedName != null) {
                    modelTensors.firstOrNull { tensor ->
                        tensor.name !in used && tensor.name == resolvedName
                    }
                } else null
            } else null

            // Strategy 2: Extract the layer name from DSL module path (ONNX-style)
            val layerName = if (byResolver == null && config.usePathBasedMatching) {
                extractLayerNameFromPath(pwp.modulePath)
            } else null

            val byName = if (byResolver == null && layerName != null) {
                modelTensors.firstOrNull { tensor ->
                    tensor.name !in used &&
                        tensor.isBias == isBiasParam &&
                        shapesCompatible(pShape, tensor.shape) &&
                        matchesLayerName(tensor.name, layerName)
                }
            } else null

            // Strategy 3: Fall back to shape-only matching
            val chosen = byResolver ?: byName ?: if (config.fallbackToShapeMatching) {
                modelTensors.firstOrNull { tensor ->
                    tensor.name !in used &&
                        tensor.isBias == isBiasParam &&
                        shapesCompatible(pShape, tensor.shape)
                }
            } else null

            if (chosen != null) {
                @Suppress("UNCHECKED_CAST")
                (param as ModuleParameter<T, V>).value = chosen.tensor
                used += chosen.name
                mapped++
                if (config.debug) {
                    val matchType = if (byName != null) "name" else "shape"
                    mappingLog += "${pwp.modulePath}/${param.name} <- ${chosen.name} ($matchType)"
                }
            } else {
                missing += "${pwp.modulePath}/${param.name} shape=${pShape}"
            }
        }

        if (config.debug) {
            println("\n=== Full Mapping ===")
            mappingLog.forEach { println("  $it") }
        }

        val unused = tensors.filter { it.name !in used }.map { "${it.name} shape=${it.shape}" }
        return MappingResult(
            mapped = mapped,
            total = paramsWithPath.size,
            missingParams = missing,
            unusedTensors = unused
        )
    }

    /**
     * Validate that all module parameters were mapped from weight tensors.
     * Throws IllegalArgumentException if mapping is incomplete.
     *
     * @param mapping The mapping result to validate
     * @param skipped List of tensors that were skipped during loading (for error message)
     */
    public fun validateAllParametersMapped(mapping: MappingResult, skipped: List<String> = emptyList()) {
        require(mapping.mapped == mapping.total) {
            buildString {
                appendLine("Only mapped ${mapping.mapped}/${mapping.total} parameters from weight tensors; aborting to avoid inconsistent weights.")
                appendList("Missing params", mapping.missingParams)
                appendList("Unused tensors", mapping.unusedTensors)
                appendList("Skipped tensors", skipped)
                appendLine("Note: Loaded weights are treated as source-of-truth. If shapes or counts differ from the DSL model, update the DSL module definition to match the model file.")
            }.trim()
        }
    }

    // ========== Name Normalization Utilities ==========

    /**
     * Extract the most specific "model.X.Y.Z" pattern from a DSL module path.
     * E.g., "Yolo8/model.22.cv2.0/model.22.cv2.0.0/Conv2d-0" -> "model.22.cv2.0.0"
     *
     * @param path The full DSL module path
     * @return The extracted layer name in normalized format, or null if not found
     */
    public fun extractLayerNameFromPath(path: String): String? {
        val parts = path.split("/")
        // Find the most specific part that looks like "model.N...."
        val modelPart = parts.lastOrNull { it.startsWith("model.") } ?: return null
        // Convert DSL naming to standard format
        return normalizeToOnnxFormat(modelPart)
    }

    /**
     * Normalize a DSL layer name to ONNX-style naming format.
     * DSL uses underscores in some places while ONNX uses dots.
     *
     * Examples:
     *   "model.18_m0_cv1" -> "model.18.m.0.cv1"
     *   "model.18_cv1" -> "model.18.cv1"
     *   "model.22.cv2.0.0" -> "model.22.cv2.0.0" (unchanged)
     *
     * @param name The DSL layer name
     * @return The normalized name
     */
    public fun normalizeToOnnxFormat(name: String): String {
        var result = name
        // Handle "_mN" -> ".m.N" pattern (bottleneck index)
        result = result.replace(Regex("_m(\\d+)")) { match ->
            ".m.${match.groupValues[1]}"
        }
        // Handle remaining underscores as dots
        result = result.replace("_", ".")
        return result
    }

    /**
     * Check if a weight tensor name matches a DSL layer name.
     * E.g., "model.22.cv2.0.0.conv.weight" matches DSL: "model.22.cv2.0.0"
     *
     * @param tensorName The name of the weight tensor
     * @param layerName The normalized DSL layer name
     * @return True if the tensor belongs to the layer
     */
    public fun matchesLayerName(tensorName: String, layerName: String?): Boolean {
        if (layerName == null) return false
        return tensorName.startsWith("$layerName.") || tensorName.startsWith("$layerName/")
    }

    // ========== Private Helpers ==========

    private fun <T : DType, V> collectParamsWithPath(module: Module<T, V>): List<ParamWithPath> {
        val out = mutableListOf<ParamWithPath>()
        fun walk(m: Module<*, *>, path: String) {
            val currentPath = if (path.isEmpty()) m.name else "$path/${m.name}"
            if (m is ModuleParameters<*, *>) {
                m.params.forEach { param ->
                    out += ParamWithPath(param, currentPath)
                }
            }
            m.modules.forEach { child ->
                walk(child, currentPath)
            }
        }
        walk(module, "")
        return out
    }

    private fun shapesCompatible(paramShape: List<Int>, tensorShape: List<Int>): Boolean {
        if (paramShape.size != tensorShape.size) return false
        return paramShape.zip(tensorShape).all { (a, b) -> a == b }
    }

    private fun StringBuilder.appendList(label: String, items: List<String>, limit: Int = 10) {
        if (items.isEmpty()) return
        appendLine("$label (${items.size}, showing up to $limit):")
        items.take(limit).forEach { append(" - ").appendLine(it) }
        if (items.size > limit) {
            appendLine(" - ... and ${items.size - limit} more")
        }
    }
}
