package sk.ainet.lang.ksp

import com.google.devtools.ksp.processing.CodeGenerator
import com.google.devtools.ksp.processing.Dependencies
import com.google.devtools.ksp.processing.KSPLogger
import com.google.devtools.ksp.symbol.KSClassDeclaration

/**
 * Generates Graph DSL extensions for interfaces annotated with @GenerateGraphDsl.
 */
class GraphDslGenerator(
    private val codeGenerator: CodeGenerator,
    private val logger: KSPLogger
) {
    fun generateGraphDsl(
        interfaceDeclaration: KSClassDeclaration,
        methods: List<MethodInfo>
    ) {
        val interfaceName = interfaceDeclaration.simpleName.asString()
        val packageName = interfaceDeclaration.packageName.asString()
        val fileName = "${interfaceName}GraphDsl"

        logger.info("Generating Graph DSL extensions for $interfaceName")

        try {
            val file = codeGenerator.createNewFile(
                dependencies = Dependencies(false, interfaceDeclaration.containingFile!!),
                packageName = "sk.ainet.lang.dag",
                fileName = fileName
            )

            file.use { outputStream ->
                val generatedCode = generateCode(packageName, fileName, methods)
                outputStream.write(generatedCode.toByteArray())
            }
        } catch (e: Exception) {
            logger.error("Failed to generate Graph DSL for $interfaceName: ${e.message}")
        }
    }

    private fun generateCode(originalPackage: String, fileName: String, methods: List<MethodInfo>): String {
        return buildString {
            appendLine("@file:Suppress(\"UnusedImport\", \"UNUSED_PARAMETER\", \"UNUSED_VARIABLE\")")
            appendLine("package sk.ainet.lang.dag")
            appendLine()
            appendLine("import sk.ainet.lang.tensor.ops.*")
            appendLine("import sk.ainet.lang.types.DType")
            appendLine("import $originalPackage.*")
            appendLine()
            appendLine("/**")
            appendLine(" * Automatically generated Graph DSL extensions.")
            appendLine(" */")
            
            for (method in methods) {
                // Skip methods that are already manually implemented and complex to generate correctly
                if (method.name in listOf("convert", "split", "op", "output", "dag", "input", "parameter", "constant")) continue
                
                // For GraphDslOps, it inherits all methods from TensorOps
                appendLine(generateMethodExtension(method))
                appendLine()
            }
        }
    }

    private fun generateMethodExtension(method: MethodInfo): String {
        return buildString {
            appendLine("/**")
            appendLine(" * DSL extension for [${method.name}].")
            appendLine(" */")
            appendLine("@DagDsl")
            
            append("public fun DagBuilder.${method.name}(")
            
            val dslParams = method.parameters.map { param ->
                val dslType = if (param.isTensor) {
                    if (param.type.endsWith("?")) "GraphValue?" else "GraphValue"
                } else if (param.type.contains("Tensor<")) {
                   if (param.type.startsWith("kotlin.collections.List<") || param.type.startsWith("List<")) "List<GraphValue>" else "GraphValue"
                } else {
                    // Replace generic type parameters with DType/Any if they appear in non-tensor params
                    param.type.replace("T", "DType").replace("V", "Any")
                }
                
                // If the parameter is optional but we don't have a default value yet,
                // we should still make it optional in the DSL if it's a Tensor/GraphValue
                val defaultPart = if (param.defaultValue != null) {
                    " = ${mapDefaultValue(param.defaultValue, param.type)}"
                } else if (param.isOptional) {
                    if (param.isTensor || param.type.contains("GraphValue")) {
                         if (dslType.endsWith("?")) " = null" else ""
                    } else if (param.type.contains("Pair<kotlin.Int, kotlin.Int>")) {
                        // Hardcode some common defaults for now since KSP can't easily extract them
                        if (param.name == "stride" || param.name == "dilation" || param.name == "kernelSize") " = 1 to 1"
                        else if (param.name == "padding") " = 0 to 0"
                        else if (param.name == "scale") " = 1 to 1"
                        else ""
                    } else if (param.type == "kotlin.Int") {
                        if (param.name == "groups") " = 1"
                        else if (param.name == "dim") " = -1"
                        else if (param.name == "startDim") " = 0"
                        else if (param.name == "endDim") " = -1"
                        else ""
                    } else if (param.type == "sk.ainet.lang.tensor.ops.UpsampleMode") {
                        " = UpsampleMode.Nearest"
                    } else if (param.type == "kotlin.Boolean") {
                        if (param.name == "alignCorners") " = false"
                        else ""
                    } else ""
                } else ""
                
                "${param.name}: $dslType$defaultPart"
            }.toMutableList()
            
            dslParams.add("id: String = \"\"")
            
            append(dslParams.joinToString(", "))
            append("): ")
            
            val returnType = if (method.returnType.isTensor) "GraphValue" else if (method.returnType.isTensorList) "List<GraphValue>" else method.returnType.type.replace("T", "DType").replace("V", "Any")
            appendLine("$returnType {")
            
            // Map parameters to attributes
            val tensorParams = method.parameters.filter { it.isTensor || it.type.contains("Tensor") }
            val nonTensorParams = method.parameters.filter { !it.isTensor && !it.type.contains("Tensor") }
            
            appendLine("    val attributes = mutableMapOf<String, Any?>()")
            for (param in nonTensorParams) {
                appendLine("    attributes[\"${param.name}\"] = ${mapAttributeValue(param)}")
            }
            
            // Special handling for hasBias in conv2d
            if (method.name == "conv2d") {
                appendLine("    attributes[\"hasBias\"] = bias != null")
            }

            // Special handling for kernel in maxPool2d (renaming kernelSize to kernel)
            if (method.name == "maxPool2d") {
                appendLine("    attributes[\"kernel\"] = attributes.remove(\"kernelSize\")")
            }

            // List of actually implemented operations in core
            val implementedOps = listOf(
                "add", "subtract", "multiply", "divide", "matmul", "transpose",
                "conv2d", "maxPool2d", "upsample2d", "reshape", "flatten",
                "relu", "softmax", "sigmoid", "squeeze", "unsqueeze"
            )

            val opClassName = if (method.name in implementedOps) {
                "${method.name.replaceFirstChar { it.uppercase() }}Operation"
            } else {
                "GenericOperation" // Fallback or we can use a string name
            }
            
            val inputList = buildString {
                append("listOfNotNull(")
                append(tensorParams.joinToString(", ") { param ->
                    if (param.type.contains("List<")) {
                        "*(${param.name}).toTypedArray()"
                    } else {
                        param.name
                    }
                })
                append(")")
            }

            val opInstantiation = if (opClassName == "GenericOperation") {
                "GenericOperation(\"${method.name}\", attributes.filterValues { it != null }.mapValues { it.value as Any })"
            } else {
                "$opClassName<sk.ainet.lang.types.DType, Any>(attributes.filterValues { it != null }.mapValues { it.value as Any })"
            }

            if (method.returnType.isTensor) {
                appendLine("    return op($opInstantiation, $inputList, id).single()")
            } else if (method.returnType.isTensorList) {
                appendLine("    return op($opInstantiation, $inputList, id)")
            } else {
                appendLine("    // TODO: Handle non-tensor return type")
                appendLine("    throw UnsupportedOperationException(\"Non-tensor return types not yet supported in generated DSL\")")
            }
            
            appendLine("}")
        }
    }

    private fun mapDefaultValue(defaultValue: String, type: String): String {
        if (type.contains("UpsampleMode")) {
            return defaultValue.replace("UpsampleMode.", "UpsampleMode.") // Just keep it
        }
        return defaultValue
    }

    private fun mapAttributeValue(param: ParameterInfo): String {
        if (param.type.contains("Pair<Int, Int>")) {
            return "listOf(${param.name}.first, ${param.name}.second)"
        }
        return param.name
    }
}
