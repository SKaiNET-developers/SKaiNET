package sk.ainet.lang.ksp

import com.google.devtools.ksp.processing.CodeGenerator
import com.google.devtools.ksp.processing.Dependencies
import com.google.devtools.ksp.processing.KSPLogger
import com.google.devtools.ksp.symbol.KSClassDeclaration

/**
 * Generates Network DSL extensions for interfaces annotated with @GenerateNetworkDsl.
 *
 * This generator creates extension functions for NeuralNetworkDsl that allow using
 * activation methods directly in the DSL builder pattern.
 */
class NetworkDslGenerator(
    private val codeGenerator: CodeGenerator,
    private val logger: KSPLogger
) {
    /**
     * Generates Network DSL extensions for the given interface.
     *
     * @param interfaceDeclaration The interface annotated with @GenerateNetworkDsl
     * @param activationMethods List of methods annotated with @ActivationDsl
     */
    fun generateNetworkDsl(
        interfaceDeclaration: KSClassDeclaration,
        activationMethods: List<MethodInfo>
    ) {
        val interfaceName = interfaceDeclaration.simpleName.asString()
        val packageName = interfaceDeclaration.packageName.asString()
        val fileName = "${interfaceName}NetworkDsl"

        logger.info("Generating Network DSL extensions for $interfaceName with ${activationMethods.size} activation methods")

        if (activationMethods.isEmpty()) {
            logger.info("No activation methods found for $interfaceName, skipping generation")
            return
        }

        try {
            val file = codeGenerator.createNewFile(
                dependencies = Dependencies(false, interfaceDeclaration.containingFile!!),
                packageName = "sk.ainet.lang.nn.dsl",
                fileName = fileName
            )

            file.use { outputStream ->
                val generatedCode = generateCode(fileName, activationMethods)
                outputStream.write(generatedCode.toByteArray())
            }

            logger.info("Successfully generated $fileName.kt")
        } catch (e: Exception) {
            logger.error("Failed to generate Network DSL for $interfaceName: ${e.message}")
        }
    }

    private fun generateCode(fileName: String, methods: List<MethodInfo>): String {
        return buildString {
            appendLine("@file:Suppress(\"UnusedImport\", \"UNUSED_PARAMETER\")")
            appendLine("package sk.ainet.lang.nn.dsl")
            appendLine()
            appendLine("import sk.ainet.lang.types.DType")
            appendLine("import sk.ainet.lang.tensor.Tensor")
            // Import extension functions for Tensor (relu, leakyRelu, elu, etc.)
            appendLine("import sk.ainet.lang.tensor.relu")
            appendLine("import sk.ainet.lang.tensor.leakyRelu")
            appendLine("import sk.ainet.lang.tensor.elu")
            appendLine("import sk.ainet.lang.tensor.sigmoid")
            appendLine("import sk.ainet.lang.tensor.silu")
            appendLine("import sk.ainet.lang.tensor.gelu")
            appendLine("import sk.ainet.lang.tensor.softmax")
            appendLine("import sk.ainet.lang.tensor.logSoftmax")
            appendLine()
            appendLine("/**")
            appendLine(" * Automatically generated Network DSL activation extensions.")
            appendLine(" * ")
            appendLine(" * These extension functions provide convenient DSL methods for common activation functions.")
            appendLine(" * Instead of writing:")
            appendLine(" * ```")
            appendLine(" * activation { it.relu() }")
            appendLine(" * ```")
            appendLine(" * You can write:")
            appendLine(" * ```")
            appendLine(" * relu()")
            appendLine(" * ```")
            appendLine(" */")
            appendLine()

            for (method in methods) {
                appendLine(generateActivationExtension(method))
                appendLine()
            }
        }
    }

    private fun generateActivationExtension(method: MethodInfo): String {
        return buildString {
            val methodName = method.name
            val tensorParams = method.parameters.filter { it.isTensor }
            val nonTensorParams = method.parameters.filter { !it.isTensor }

            // Skip methods that aren't activation-like (should have 1 tensor param and return tensor)
            if (tensorParams.size != 1 || !method.returnType.isTensor) {
                appendLine("// Skipped ${method.name}: not an activation-like method")
                return@buildString
            }

            appendLine("/**")
            appendLine(" * Adds a $methodName activation layer to the network.")
            if (nonTensorParams.isNotEmpty()) {
                appendLine(" *")
                for (param in nonTensorParams) {
                    appendLine(" * @param ${param.name} ${getParamDescription(param)}")
                }
            }
            appendLine(" * @param id Optional identifier for the layer")
            appendLine(" */")
            appendLine("@NetworkDsl")

            // Build parameter list
            val paramList = buildList {
                for (param in nonTensorParams) {
                    val defaultValue = getDefaultValue(param)
                    val simpleType = simplifyType(param.type)
                    if (defaultValue != null) {
                        add("${param.name}: $simpleType = $defaultValue")
                    } else {
                        add("${param.name}: $simpleType")
                    }
                }
                add("id: String = \"\"")
            }.joinToString(", ")

            appendLine("public fun <T : DType, V> NeuralNetworkDsl<T, V>.$methodName($paramList) {")

            // Generate the activation call
            val activationArgs = if (nonTensorParams.isEmpty()) {
                "it.$methodName()"
            } else {
                val args = nonTensorParams.joinToString(", ") { it.name }
                "it.$methodName($args)"
            }

            appendLine("    activation(id) { $activationArgs }")
            appendLine("}")
        }
    }

    private fun simplifyType(type: String): String {
        return when {
            type == "kotlin.Float" -> "Float"
            type == "kotlin.Int" -> "Int"
            type == "kotlin.Double" -> "Double"
            type == "kotlin.Boolean" -> "Boolean"
            type == "kotlin.String" -> "String"
            else -> type
        }
    }

    private fun getDefaultValue(param: ParameterInfo): String? {
        // Handle known default values for activation parameters
        return when {
            param.name == "negativeSlope" && param.type.contains("Float") -> "0.01f"
            param.name == "alpha" && param.type.contains("Float") -> "1.0f"
            param.name == "dim" && param.type.contains("Int") -> "-1"
            param.defaultValue != null -> param.defaultValue
            param.isOptional -> when {
                param.type.contains("Float") -> "0.0f"
                param.type.contains("Int") -> "0"
                param.type.contains("Double") -> "0.0"
                param.type.contains("Boolean") -> "false"
                else -> null
            }
            else -> null
        }
    }

    private fun getParamDescription(param: ParameterInfo): String {
        return when (param.name) {
            "negativeSlope" -> "Controls the angle of the negative slope in LeakyReLU"
            "alpha" -> "Scale for the negative factor in ELU"
            "dim" -> "Dimension along which to compute the operation"
            else -> "Parameter for the activation function"
        }
    }
}
