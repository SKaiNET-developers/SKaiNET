package sk.ainet.lang.ksp

import com.google.devtools.ksp.symbol.*

/**
 * Data model representing analyzed method information for code generation.
 */
data class MethodInfo(
    val name: String,
    val parameters: List<ParameterInfo>,
    val returnType: ReturnTypeInfo,
    val typeParameters: List<String>
)

/**
 * Data model representing parameter information.
 */
data class ParameterInfo(
    val name: String,
    val type: String,
    val isTensor: Boolean,
    val isOptional: Boolean,
    val defaultValue: String?
)

/**
 * Data model representing return type information.
 */
data class ReturnTypeInfo(
    val type: String,
    val isTensor: Boolean,
    val isTensorList: Boolean
)

/**
 * Represents different strategies for handling operation attributes.
 */
sealed class AttributeStrategy {
    /** Use an existing OpAttributeFactory method */
    data class UseFactory(val methodCall: String) : AttributeStrategy()
    
    /** Use default parameter-name mapping */
    data class DefaultMapping(val parameters: List<ParameterInfo>) : AttributeStrategy()
    
    /** Use custom mapping with additional logic */
    data class CustomMapping(val factoryMethod: String, val additionalParams: List<ParameterInfo>) : AttributeStrategy()
    
    /** No attributes needed */
    object NoAttributes : AttributeStrategy()
}

/**
 * Analyzes method signatures from KSP declarations to extract information
 * needed for generating tracing wrapper implementations.
 */
class MethodAnalyzer {

    /**
     * Analyzes all methods in the given interface declaration.
     * 
     * @param interfaceDeclaration The interface to analyze
     * @return List of analyzed method information
     */
    fun analyzeInterface(interfaceDeclaration: KSClassDeclaration): List<MethodInfo> {
        return interfaceDeclaration.getAllFunctions()
            .filter { it.isAbstract } // Only abstract methods need implementation
            .map { analyzeMethod(it) }
            .toList()
    }

    /**
     * Analyzes a single method declaration.
     * 
     * @param method The method to analyze
     * @return Analyzed method information
     */
    fun analyzeMethod(method: KSFunctionDeclaration): MethodInfo {
        val name = method.simpleName.asString()
        val parameters = method.parameters.map { analyzeParameter(it) }
        val returnType = analyzeReturnType(method.returnType?.resolve())
        val typeParameters = method.typeParameters.map { analyzeTypeParameter(it) }

        return MethodInfo(
            name = name,
            parameters = parameters,
            returnType = returnType,
            typeParameters = typeParameters
        )
    }

    /**
     * Analyzes a method parameter.
     * 
     * @param parameter The parameter to analyze
     * @return Analyzed parameter information
     */
    private fun analyzeParameter(parameter: KSValueParameter): ParameterInfo {
        val name = parameter.name?.asString() ?: "unnamed"
        val type = parameter.type.resolve()
        val typeString = getFullTypeString(type)
        val isTensor = isTensorType(type)
        val isOptional = parameter.hasDefault || type.isMarkedNullable
        val defaultValue = if (parameter.hasDefault) extractDefaultValue(parameter) else null

        return ParameterInfo(
            name = name,
            type = typeString,
            isTensor = isTensor,
            isOptional = isOptional,
            defaultValue = defaultValue
        )
    }

    /**
     * Analyzes a type parameter declaration.
     * 
     * @param typeParameter The type parameter to analyze
     * @return Type parameter string with bounds
     */
    private fun analyzeTypeParameter(typeParameter: KSTypeParameter): String {
        val name = typeParameter.name.asString()
        val bounds = typeParameter.bounds.toList()
        
        // Filter out implicit kotlin.Any bounds
        val explicitBounds = bounds.filter { bound ->
            val resolvedBound = bound.resolve()
            val qualifiedName = resolvedBound.declaration.qualifiedName?.asString()
            qualifiedName != "kotlin.Any"
        }
        
        return if (explicitBounds.isNotEmpty()) {
            val boundsString = explicitBounds.joinToString(" & ") { bound ->
                val resolvedBound = bound.resolve()
                val qualifiedName = resolvedBound.declaration.qualifiedName?.asString()
                // Use simple name for common types
                when (qualifiedName) {
                    "sk.ainet.lang.types.DType" -> "DType"
                    else -> qualifiedName ?: bound.toString()
                }
            }
            "$name : $boundsString"
        } else {
            name
        }
    }

    /**
     * Analyzes a method return type.
     * 
     * @param returnType The return type to analyze, null for Unit
     * @return Analyzed return type information
     */
    private fun analyzeReturnType(returnType: KSType?): ReturnTypeInfo {
        if (returnType == null) {
            return ReturnTypeInfo(
                type = "Unit",
                isTensor = false,
                isTensorList = false
            )
        }

        val typeString = getFullTypeString(returnType)
        val isTensor = isTensorType(returnType)
        val isTensorList = isTensorListType(returnType)

        return ReturnTypeInfo(
            type = typeString,
            isTensor = isTensor,
            isTensorList = isTensorList
        )
    }

    /**
     * Determines if a type is a Tensor type.
     * 
     * @param type The type to check
     * @return true if the type is a Tensor<T, V>
     */
    private fun isTensorType(type: KSType): Boolean {
        val declaration = type.declaration
        val qualifiedName = declaration.qualifiedName?.asString()
        
        // Check if it's sk.ainet.lang.tensor.Tensor
        return qualifiedName == "sk.ainet.lang.tensor.Tensor"
    }

    /**
     * Determines if a type is a List of Tensor types.
     * 
     * @param type The type to check
     * @return true if the type is List<Tensor<T, V>>
     */
    private fun isTensorListType(type: KSType): Boolean {
        val declaration = type.declaration
        val qualifiedName = declaration.qualifiedName?.asString()
        
        // Check if it's a List
        if (qualifiedName != "kotlin.collections.List") {
            return false
        }

        // Check if the type argument is a Tensor
        val typeArguments = type.arguments
        if (typeArguments.size != 1) {
            return false
        }

        val firstArgument = typeArguments.first()
        val argumentType = firstArgument.type?.resolve()
        return argumentType?.let { isTensorType(it) } ?: false
    }

    /**
     * Extracts the default value from a parameter if available.
     * This is a simplified implementation - KSP doesn't provide easy access to default values.
     * 
     * @param parameter The parameter with a default value
     * @return String representation of the default value, or null if not extractable
     */
    private fun extractDefaultValue(parameter: KSValueParameter): String? {
        // KSP doesn't provide direct access to default values in the AST
        // For now, we'll return null and handle defaults in the generated code
        // by using the same parameter signature
        return null
    }

    /**
     * Gets the full type string including generic parameters for code generation.
     * 
     * @param type The KSType to convert to string
     * @return Full type string suitable for code generation
     */
    fun getFullTypeString(type: KSType): String {
        val declaration = type.declaration
        
        // Handle type parameters first - they should use simple names
        if (declaration is KSTypeParameter) {
            return declaration.simpleName.asString()
        }
        
        val qualifiedName = declaration.qualifiedName?.asString() ?: declaration.simpleName.asString()
        
        val typeArguments = type.arguments
        if (typeArguments.isEmpty()) {
            return qualifiedName
        }

        val argumentStrings = typeArguments.map { argument ->
            when (argument.variance) {
                Variance.STAR -> "*"
                else -> {
                    val argType = argument.type?.resolve()
                    if (argType != null) {
                        // For type parameters, use simple name instead of fully qualified
                        val argDeclaration = argType.declaration
                        if (argDeclaration is KSTypeParameter) {
                            argDeclaration.simpleName.asString()
                        } else {
                            getFullTypeString(argType)
                        }
                    } else {
                        "Any?"
                    }
                }
            }
        }

        return "$qualifiedName<${argumentStrings.joinToString(", ")}>"
    }

    /**
     * Gets the simple type name without package qualification.
     * 
     * @param type The KSType to get simple name for
     * @return Simple type name
     */
    fun getSimpleTypeName(type: KSType): String {
        return type.declaration.simpleName.asString()
    }
    
    /**
     * Determines the appropriate attribute strategy for a method.
     * This analyzes the method signature to decide how to handle attributes.
     */
    fun determineAttributeStrategy(method: MethodInfo): AttributeStrategy {
        val tensorParams = method.parameters.filter { it.isTensor }
        val nonTensorParams = method.parameters.filter { !it.isTensor }
        
        // Check for specific OpAttributeFactory methods
        val factoryMethod = when {
            // Scalar operations
            method.name in listOf("addScalar", "subScalar", "mulScalar", "divScalar") && 
            tensorParams.size == 1 && method.returnType.isTensor -> {
                val tensor = tensorParams[0].name
                val scalar = nonTensorParams.find { it.type.contains("Number") }?.name ?: "b"
                "scalarOp($tensor as Tensor<*, *>, $scalar, result as Tensor<*, *>, false)"
            }
            
            // Reversed scalar operations
            method.name in listOf("rsubScalar", "rdivScalar") && 
            tensorParams.size == 1 && method.returnType.isTensor -> {
                val tensor = tensorParams[0].name
                val scalar = nonTensorParams.find { it.type.contains("Number") }?.name ?: "a"
                "scalarOp($tensor as Tensor<*, *>, $scalar, result as Tensor<*, *>, true)"
            }
            
            // Unary operations
            method.name in listOf("relu", "sigmoid", "tanh", "exp", "log", "abs", "neg", "silu", "softmax", "logSoftmax", "sqrt") && 
            tensorParams.size == 1 && method.returnType.isTensor -> {
                val param = tensorParams[0].name
                "unary($param as Tensor<*, *>, result as Tensor<*, *>)"
            }
            
            // Conv2d operation
            method.name == "conv2d" && tensorParams.size >= 2 && method.returnType.isTensor -> {
                generateConv2dCall(method)
            }

            // Shape operations - use shapesAndDTypes
            method.name in listOf("reshape", "flatten", "squeeze", "unsqueeze", "transpose", "concat", "split") -> {
                val inputs = tensorParams.map { "${it.name} as Tensor<*, *>" }.joinToString(", ")
                val outputs = if (method.returnType.isTensorList) "result as List<Tensor<*, *>>" else "listOf(result as Tensor<*, *>)"
                "shapesAndDTypes(listOfNotNull($inputs), $outputs)"
            }
            
            else -> null
        }
        
        return when {
            factoryMethod != null && nonTensorParams.isNotEmpty() -> {
                // Use factory method and add additional parameters
                AttributeStrategy.CustomMapping(factoryMethod, nonTensorParams)
            }
            factoryMethod != null -> {
                // Use factory method only
                AttributeStrategy.UseFactory(factoryMethod)
            }
            nonTensorParams.isNotEmpty() -> {
                // Use default parameter mapping
                AttributeStrategy.DefaultMapping(nonTensorParams)
            }
            else -> {
                // No attributes needed
                AttributeStrategy.NoAttributes
            }
        }
    }
    
    /**
     * Generates the conv2d factory method call.
     */
    private fun generateConv2dCall(method: MethodInfo): String? {
        val tensorParams = method.parameters.filter { it.isTensor }
        if (tensorParams.size < 2) return null
        
        val input = tensorParams[0].name
        val weight = tensorParams[1].name
        val bias = if (tensorParams.size > 2) tensorParams[2].name else "null"
        
        // Look for stride, padding, dilation, groups parameters
        val stride = method.parameters.find { it.name == "stride" }?.name ?: "Pair(1, 1)"
        val padding = method.parameters.find { it.name == "padding" }?.name ?: "Pair(0, 0)"
        val dilation = method.parameters.find { it.name == "dilation" }?.name ?: "Pair(1, 1)"
        val groups = method.parameters.find { it.name == "groups" }?.name ?: "1"
        
        // Add type casts to Tensor<*, *> for OpAttributeFactory compatibility
        val inputCast = "$input as Tensor<*, *>"
        val weightCast = "$weight as Tensor<*, *>"
        val biasCast = if (bias == "null") "null" else "$bias as Tensor<*, *>?"
        val resultCast = "result as Tensor<*, *>"
        
        return "conv2d($inputCast, $weightCast, $biasCast, $resultCast, $stride, $padding, $dilation, $groups)"
    }
}