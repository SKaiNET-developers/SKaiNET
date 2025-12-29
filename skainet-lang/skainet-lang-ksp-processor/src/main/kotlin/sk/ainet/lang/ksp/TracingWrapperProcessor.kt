package sk.ainet.lang.ksp

import com.google.devtools.ksp.processing.*
import com.google.devtools.ksp.symbol.*
import com.google.devtools.ksp.validate

/**
 * Exception thrown when code generation fails due to validation or generation errors.
 */
class CodeGenerationException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Generates tracing wrapper class code for interfaces annotated with @GenerateTracingWrapper.
 * 
 * This class handles the actual code generation logic, producing complete tracing
 * implementations that delegate to base implementations while emitting OpTrace events.
 */
class TracingWrapperGenerator(
    private val codeGenerator: CodeGenerator,
    private val logger: KSPLogger
) {
    
    /**
     * Generates a complete tracing wrapper class for the given interface.
     */
    fun generateTracingWrapper(
        interfaceDeclaration: KSClassDeclaration,
        methods: List<MethodInfo>
    ) {
        val interfaceName = interfaceDeclaration.simpleName.asString()
        val packageName = interfaceDeclaration.packageName.asString()
        val generatedClassName = "Ksp$interfaceName"
        
        logger.info("Generating class $generatedClassName with ${methods.size} methods")
        
        try {
            // Validate that we can generate code for all methods
            validateCodeGeneration(methods, interfaceName)
            
            // Create the output file
            val file = codeGenerator.createNewFile(
                dependencies = Dependencies(false, interfaceDeclaration.containingFile!!),
                packageName = packageName,
                fileName = generatedClassName
            )
            
            file.use { outputStream ->
                val generatedCode = generateClassCode(packageName, generatedClassName, interfaceName, methods)
                
                // Validate the generated code before writing
                validateGeneratedCode(generatedCode, generatedClassName)
                
                outputStream.write(generatedCode)
            }
            
            logger.info("Successfully generated $generatedClassName.kt in package $packageName")
            
        } catch (e: CodeGenerationException) {
            logger.error("Code generation failed for $interfaceName: ${e.message}", interfaceDeclaration)
            throw e
        } catch (e: Exception) {
            logger.error("Unexpected error during code generation for $interfaceName: ${e.message}", interfaceDeclaration)
            throw CodeGenerationException("Failed to generate tracing wrapper for $interfaceName", e)
        }
    }
    
    /**
     * Validates that code generation is possible for all methods.
     */
    private fun validateCodeGeneration(methods: List<MethodInfo>, interfaceName: String) {
        val issues = mutableListOf<String>()
        
        for (method in methods) {
            // Check for method name conflicts with generated code
            if (method.name in listOf("base", "sink", "session")) {
                issues.add("Method '${method.name}' conflicts with generated constructor parameters")
            }
            
            // Check for excessively complex method signatures
            if (method.parameters.size > 20) {
                issues.add("Method '${method.name}' has too many parameters (${method.parameters.size}), maximum is 20")
            }
            
            // Check for unsupported parameter combinations
            val tensorParams = method.parameters.filter { it.isTensor }
            val nonTensorParams = method.parameters.filter { !it.isTensor }
            
            if (tensorParams.size > 10) {
                issues.add("Method '${method.name}' has too many tensor parameters (${tensorParams.size}), maximum is 10")
            }
            
            // Validate attribute generation is possible
            try {
                val methodAnalyzer = MethodAnalyzer()
                methodAnalyzer.determineAttributeStrategy(method)
            } catch (e: Exception) {
                issues.add("Cannot determine attribute strategy for method '${method.name}': ${e.message}")
            }
        }
        
        if (issues.isNotEmpty()) {
            val errorMessage = "Code generation validation failed for $interfaceName:\n" +
                issues.joinToString("\n") { "  - $it" }
            throw CodeGenerationException(errorMessage)
        }
    }
    
    /**
     * Validates the generated code for basic correctness.
     */
    private fun validateGeneratedCode(generatedCode: ByteArray, className: String) {
        val codeString = String(generatedCode)
        val issues = mutableListOf<String>()
        
        // Check for basic syntax issues
        if (!codeString.contains("class $className")) {
            issues.add("Generated code does not contain expected class declaration")
        }
        
        if (!codeString.contains("override fun")) {
            issues.add("Generated code does not contain any method implementations")
        }
        
        // Check for balanced braces
        val openBraces = codeString.count { it == '{' }
        val closeBraces = codeString.count { it == '}' }
        if (openBraces != closeBraces) {
            issues.add("Unbalanced braces in generated code (open: $openBraces, close: $closeBraces)")
        }
        
        // Check for balanced parentheses
        val openParens = codeString.count { it == '(' }
        val closeParens = codeString.count { it == ')' }
        if (openParens != closeParens) {
            issues.add("Unbalanced parentheses in generated code (open: $openParens, close: $closeParens)")
        }
        
        // Check for required imports
        val requiredImports = listOf("OpSink", "OpTrace", "TraceSession", "TensorRef")
        for (import in requiredImports) {
            if (!codeString.contains(import)) {
                issues.add("Missing required import or usage: $import")
            }
        }
        
        // Check for proper package declaration
        if (!codeString.startsWith("package ")) {
            issues.add("Generated code does not start with package declaration")
        }
        
        if (issues.isNotEmpty()) {
            val errorMessage = "Generated code validation failed for $className:\n" +
                issues.joinToString("\n") { "  - $it" }
            throw CodeGenerationException(errorMessage)
        }
        
        logger.info("Generated code validation passed for $className")
    }
    
    /**
     * Generates the complete class code as a byte array.
     */
    private fun generateClassCode(
        packageName: String,
        className: String,
        interfaceName: String,
        methods: List<MethodInfo>
    ): ByteArray {
        val code = buildString {
            // Package declaration
            appendLine("package $packageName")
            appendLine()
            
            // Imports
            appendLine("import sk.ainet.lang.trace.OpSink")
            appendLine("import sk.ainet.lang.trace.OpTrace")
            appendLine("import sk.ainet.lang.trace.TraceSession")
            appendLine("import sk.ainet.lang.trace.TensorRef")
            appendLine("import sk.ainet.lang.trace.OpAttributeFactory")
            appendLine("import sk.ainet.lang.types.DType")
            if (methods.any { it.returnType.isTensorList || it.parameters.any { param -> param.isTensor } }) {
                appendLine("import sk.ainet.lang.tensor.Tensor")
            }
            appendLine()
            
            // Class declaration
            appendLine("/**")
            appendLine(" * Generated tracing wrapper for $interfaceName.")
            appendLine(" * This class delegates all method calls to the base implementation")
            appendLine(" * while emitting OpTrace events to capture the computation graph.")
            appendLine(" */")
            appendLine("public class $className(")
            appendLine("    private val base: $interfaceName,")
            appendLine("    private val sink: OpSink,")
            appendLine("    private val session: TraceSession = TraceSession()")
            appendLine(") : $interfaceName {")
            appendLine()
            
            // Add wrap method for ensuring recursive tracing
            appendLine("    @Suppress(\"UNCHECKED_CAST\")")
            appendLine("    private fun <T : DType, V, R> wrap(result: R): R {")
            appendLine("        return when (result) {")
            appendLine("            is Tensor<*, *> -> {")
            appendLine("                if (result.ops === this) result as R")
            appendLine("                else object : Tensor<T, V> by (result as Tensor<T, V>) {")
            appendLine("                    override val ops: TensorOps get() = this@$className")
            appendLine("                    override fun toString(): String = result.toString()")
            appendLine("                } as R")
            appendLine("            }")
            appendLine("            is List<*> -> result.map { wrap<DType, Any?, Any?>(it) } as R")
            appendLine("            else -> result")
            appendLine("        }")
            appendLine("    }")
            appendLine()
            
            // Generate method implementations
            methods.forEach { method ->
                appendLine(generateMethodImplementation(method))
                appendLine()
            }
            
            appendLine("}")
        }
        
        return code.toByteArray()
    }
    
    /**
     * Generates the implementation for a single method.
     */
    private fun generateMethodImplementation(method: MethodInfo): String {
        return buildString {
            // Method signature
            append("    override fun ")
            if (method.typeParameters.isNotEmpty()) {
                append("<${method.typeParameters.joinToString(", ")}> ")
            }
            append("${method.name}(")
            append(method.parameters.joinToString(", ") { param ->
                // Fix the type for nullable parameters that should have ? suffix
                val correctedType = when {
                    param.name == "dim" && method.name in listOf("squeeze", "sum", "mean", "variance") && !param.type.endsWith("?") -> "${param.type}?"
                    param.name == "bias" && method.name == "conv2d" && !param.type.endsWith("?") -> "${param.type}?"
                    else -> param.type
                }
                
                // Override methods cannot specify default values - they inherit them from the interface
                "${param.name}: $correctedType"
            })
            appendLine("): ${method.returnType.type} {")
            
            // Method body - delegate to base and trace
            appendLine("        // Delegate to base implementation")
            val paramNames = method.parameters.map { it.name }
            val paramList = paramNames.joinToString(", ")
            
            if (method.returnType.type == "Unit" || method.returnType.type == "kotlin.Unit") {
                // For Unit return type, call base method directly
                appendLine("        base.${method.name}($paramList)")
            } else {
                // For non-Unit return type, capture and wrap the result
                appendLine("        val result = wrap<DType, Any?, ${method.returnType.type}>(base.${method.name}($paramList))")
            }
            appendLine()
            
            // Generate tracing logic
            appendLine("        // Capture input tensor references")
            val tensorParams = method.parameters.filter { it.isTensor }
            if (tensorParams.isNotEmpty()) {
                val inputRefs = tensorParams.map { param ->
                    if (param.type.endsWith("?")) {
                        // Handle nullable tensor parameters
                        "if (${param.name} != null) session.refOf(${param.name} as Tensor<*, *>) else null"
                    } else {
                        "session.refOf(${param.name} as Tensor<*, *>)"
                    }
                }
                appendLine("        val inputs = listOfNotNull(${inputRefs.joinToString(", ")})")
            } else if (method.returnType.isTensor || method.returnType.isTensorList) {
                // If there are no tensor parameters but it returns a tensor, we still need inputs for OpTrace
                // although it might be an empty list.
                appendLine("        val inputs = emptyList<TensorRef>()")
            } else {
                appendLine("        val inputs = emptyList<TensorRef>()")
            }
            appendLine()
            
            // Capture output tensor references
            appendLine("        // Capture output tensor references")
            when {
                method.returnType.isTensorList -> {
                    // Handle List<Tensor<T, V>> return type
                    appendLine("        val outputs = session.refsOf(result as List<Tensor<*, *>>)")
                }
                method.returnType.isTensor -> {
                    // Handle single Tensor<T, V> return type
                    appendLine("        val outputs = listOf(session.refOf(result as Tensor<*, *>))")
                }
                else -> {
                    // Non-tensor return type
                    appendLine("        val outputs = emptyList<TensorRef>()")
                }
            }
            appendLine()

            val anyTensorParams = tensorParams.isNotEmpty() || method.returnType.isTensorList
            val hasTensorResult = method.returnType.isTensor || method.returnType.isTensorList
            
            // Emit OpTrace only if tensors are involved
            if (anyTensorParams || hasTensorResult) {
                // Generate attributes using strategy pattern
                appendLine("        // Generate operation attributes")
                val methodAnalyzer = MethodAnalyzer()
                val attributeStrategy = methodAnalyzer.determineAttributeStrategy(method)
                
                when (attributeStrategy) {
                    is AttributeStrategy.UseFactory -> {
                        appendLine("        val combinedAttrs = OpAttributeFactory.${attributeStrategy.methodCall}")
                    }
                    
                    is AttributeStrategy.DefaultMapping -> {
                        val attrMapping = generateDefaultAttributeMapping(attributeStrategy.parameters)
                        appendLine("        val combinedAttrs = $attrMapping")
                    }
                    
                    is AttributeStrategy.CustomMapping -> {
                        appendLine("        val attrs = OpAttributeFactory.${attributeStrategy.factoryMethod}")
                        val additionalAttrs = generateDefaultAttributeMapping(attributeStrategy.additionalParams)
                        appendLine("        val additionalAttrs = $additionalAttrs")
                        appendLine("        val combinedAttrs = attrs + additionalAttrs")
                    }
                    
                    is AttributeStrategy.NoAttributes -> {
                        // Use shapesAndDTypes if we have tensors, otherwise empty map
                        if (tensorParams.isNotEmpty() && method.returnType.isTensor) {
                            val inputTensors = tensorParams.map { param ->
                                if (param.type.endsWith("?")) "${param.name} as Tensor<*, *>?" else "${param.name} as Tensor<*, *>"
                            }.joinToString(", ")
                            val outputTensors = if (method.returnType.isTensorList) "result as List<Tensor<*, *>>" else "listOf(result as Tensor<*, *>)"
                            appendLine("        val combinedAttrs = OpAttributeFactory.shapesAndDTypes(listOfNotNull($inputTensors), $outputTensors)")
                        } else {
                            appendLine("        val combinedAttrs = emptyMap<String, Any?>()")
                        }
                    }
                }
                appendLine()
                
                // Emit OpTrace
                appendLine("        // Emit OpTrace")
                appendLine("        sink.onOpExecuted(OpTrace(\"${method.name}\", inputs, outputs, combinedAttrs))")
                appendLine()
            }
            
            // Return the result if not Unit
            if (method.returnType.type != "Unit" && method.returnType.type != "kotlin.Unit") {
                appendLine("        return result")
            }
            appendLine("    }")
        }
    }
    
    /**
     * Determines if there's an appropriate OpAttributeFactory method for this operation.
     * This method detects existing OpAttributeFactory methods by operation name and parameter patterns.
     */
    private fun getOpAttributeFactoryMethod(method: MethodInfo): String? {
        val tensorParams = method.parameters.filter { it.isTensor }
        
        return when {
            // Binary operations: two tensor inputs, one tensor output
            method.name in listOf("add", "subtract", "multiply", "divide", "matmul") && 
            tensorParams.size == 2 && method.returnType.isTensor -> {
                val param1 = tensorParams[0].name
                val param2 = tensorParams[1].name
                "binary($param1, $param2, result)"
            }
            
            // Unary operations: one tensor input, one tensor output
            method.name in listOf("relu", "sigmoid", "tanh", "exp", "log", "abs", "neg", "silu", "softmax", "logSoftmax") && 
            tensorParams.size == 1 && method.returnType.isTensor -> {
                val param = tensorParams[0].name
                "unary($param, result)"
            }
            
            // Conv2d operation: specific parameter pattern
            method.name == "conv2d" && tensorParams.size >= 2 && method.returnType.isTensor -> {
                generateConv2dAttributeCall(method)
            }
            
            else -> null
        }
    }
    
    /**
     * Generates the OpAttributeFactory.conv2d call for conv2d operations.
     */
    private fun generateConv2dAttributeCall(method: MethodInfo): String? {
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
        
        return "conv2d($input, $weight, $bias, result, $stride, $padding, $dilation, $groups)"
    }
    
    /**
     * Generates default attribute mapping for non-tensor parameters.
     * Handles primitive types, collections, and custom data classes.
     */
    private fun generateDefaultAttributeMapping(nonTensorParams: List<ParameterInfo>): String {
        val attrEntries = nonTensorParams.map { param ->
            val value = generateAttributeValue(param)
            "\"${param.name}\" to $value"
        }
        return "mapOf(${attrEntries.joinToString(", ")})"
    }
    
    /**
     * Generates the appropriate attribute value expression for a parameter.
     * Handles different parameter types including primitives, collections, and custom classes.
     */
    private fun generateAttributeValue(param: ParameterInfo): String {
        return when {
            // Handle primitive types directly
            param.type in listOf("kotlin.Int", "kotlin.Float", "kotlin.Double", "kotlin.Boolean", 
                                "kotlin.String", "kotlin.Long", "kotlin.Short", "kotlin.Byte") -> {
                param.name
            }
            
            // Handle Pair types - convert to list for serialization
            param.type.startsWith("kotlin.Pair") -> {
                "listOf(${param.name}.first, ${param.name}.second)"
            }
            
            // Handle Triple types - convert to list for serialization
            param.type.startsWith("kotlin.Triple") -> {
                "listOf(${param.name}.first, ${param.name}.second, ${param.name}.third)"
            }
            
            // Handle List types directly
            param.type.startsWith("kotlin.collections.List") -> {
                param.name
            }
            
            // Handle Array types - convert to list
            param.type.endsWith("Array") -> {
                "${param.name}.toList()"
            }
            
            // Handle enum types - use name property
            isEnumType(param.type) -> {
                "${param.name}.name"
            }
            
            // Handle nullable types
            param.type.endsWith("?") -> {
                param.name
            }
            
            // For custom data classes and other complex types, use toString()
            else -> {
                "${param.name}.toString()"
            }
        }
    }
    
    /**
     * Determines if a type is likely an enum type based on naming conventions.
     * This is a heuristic since KSP type information might not always be complete.
     */
    private fun isEnumType(typeName: String): Boolean {
        // Common enum naming patterns in the codebase
        return typeName.contains("Mode") || 
               typeName.contains("Type") || 
               typeName.contains("Strategy") ||
               typeName.endsWith("Enum")
    }
}

/**
 * KSP processor that generates tracing wrapper implementations for interfaces
 * annotated with @GenerateTracingWrapper.
 * 
 * This processor analyzes TensorOps interfaces and generates complete tracing
 * wrappers that delegate to base implementations while emitting OpTrace events
 * to capture the computation graph.
 */
class TracingWrapperProcessor(
    private val codeGenerator: CodeGenerator,
    private val logger: KSPLogger
) : SymbolProcessor {

    override fun process(resolver: Resolver): List<KSAnnotated> {
        logger.info("Starting TracingWrapperProcessor...")

        // Validate required dependencies are available
        if (!validateDependencies(resolver)) {
            logger.error("Required dependencies are missing. Cannot proceed with code generation.")
            return emptyList()
        }

        // Find interfaces annotated with @GenerateTracingWrapper
        val annotatedSymbols = resolver
            .getSymbolsWithAnnotation("sk.ainet.lang.trace.GenerateTracingWrapper")
            .filterIsInstance<KSClassDeclaration>()
            .filter { it.validate() }
            .toList()

        if (annotatedSymbols.isEmpty()) {
            logger.info("No interfaces annotated with @GenerateTracingWrapper found")
            return emptyList()
        }

        logger.info("Found ${annotatedSymbols.size} annotated interfaces")

        // Process each annotated interface
        val unprocessedSymbols = mutableListOf<KSAnnotated>()
        
        for (symbol in annotatedSymbols) {
            try {
                if (validateAnnotatedSymbol(symbol)) {
                    generateTracingWrapper(symbol)
                } else {
                    // Add to unprocessed if validation fails
                    unprocessedSymbols.add(symbol)
                }
            } catch (e: Exception) {
                logger.error("Failed to process interface ${symbol.simpleName.asString()}: ${e.message}", symbol)
                unprocessedSymbols.add(symbol)
            }
        }

        return unprocessedSymbols
    }

    /**
     * Validates that required dependencies are available in the classpath.
     * This ensures that all necessary classes for code generation are present.
     */
    private fun validateDependencies(resolver: Resolver): Boolean {
        // Skip dependency validation during KSP processing since the tracing classes
        // are in a different module that would create circular dependencies.
        // The generated code will be validated at compile time when all dependencies are available.
        logger.info("Skipping dependency validation - tracing classes will be resolved at compile time")
        return true
    }

    /**
     * Validates that the annotated symbol meets all requirements for code generation.
     * This includes checking annotation target, interface structure, and method signatures.
     */
    private fun validateAnnotatedSymbol(symbol: KSClassDeclaration): Boolean {
        val symbolName = symbol.simpleName.asString()
        
        // Check if it's an interface
        if (symbol.classKind != ClassKind.INTERFACE) {
            logger.error(
                "GenerateTracingWrapper can only be applied to interfaces, " +
                "but $symbolName is a ${symbol.classKind.name.lowercase()}. " +
                "Please apply the annotation only to interface declarations.",
                symbol
            )
            return false
        }

        // Check if the interface has methods to process
        val methods = symbol.getAllFunctions().filter { it.isAbstract }.toList()
        if (methods.isEmpty()) {
            logger.warn(
                "Interface $symbolName has no abstract methods to trace. " +
                "The generated wrapper will be empty.",
                symbol
            )
        }

        // Validate each method signature
        val methodAnalyzer = MethodAnalyzer()
        for (method in methods) {
            if (!validateMethodSignature(method, methodAnalyzer)) {
                logger.error(
                    "Interface $symbolName contains unsupported method signatures. " +
                    "Code generation aborted.",
                    symbol
                )
                return false
            }
        }

        // Check for package accessibility
        val packageName = symbol.packageName.asString()
        if (packageName.isEmpty()) {
            logger.error(
                "Interface $symbolName is in the default package. " +
                "Generated classes must be in a named package.",
                symbol
            )
            return false
        }

        return true
    }

    /**
     * Validates that a method signature is supported for code generation.
     * This checks parameter types, return types, and other constraints.
     */
    private fun validateMethodSignature(method: KSFunctionDeclaration, analyzer: MethodAnalyzer): Boolean {
        val methodName = method.simpleName.asString()
        
        try {
            // Attempt to analyze the method - this will catch most issues
            val methodInfo = analyzer.analyzeMethod(method)
            
            // Validate return type is supported
            if (!validateReturnType(method.returnType?.resolve(), methodName)) {
                return false
            }
            
            // Validate all parameters are supported
            for (param in method.parameters) {
                if (!validateParameterType(param, methodName)) {
                    return false
                }
            }
            
            // Validate type parameters are reasonable
            if (method.typeParameters.size > 5) {
                logger.error(
                    "Method $methodName has too many type parameters (${method.typeParameters.size}). " +
                    "Maximum supported is 5 type parameters.",
                    method
                )
                return false
            }
            
            return true
            
        } catch (e: Exception) {
            logger.error(
                "Failed to analyze method signature for $methodName: ${e.message}. " +
                "This method signature is not supported for tracing wrapper generation.",
                method
            )
            return false
        }
    }

    /**
     * Validates that a return type is supported for code generation.
     */
    private fun validateReturnType(returnType: KSType?, methodName: String): Boolean {
        if (returnType == null) {
            // Unit return type is always supported
            return true
        }
        
        val typeName = returnType.declaration.qualifiedName?.asString() ?: returnType.toString()
        
        // Check for unsupported generic wildcards
        if (typeName.contains("*") || typeName.contains("?") && !returnType.isMarkedNullable) {
            logger.error(
                "Method $methodName has unsupported return type with wildcards: $typeName. " +
                "Wildcard types are not supported in generated tracing wrappers."
            )
            return false
        }
        
        // Check for excessively nested generics
        val genericDepth = typeName.count { it == '<' }
        if (genericDepth > 3) {
            logger.error(
                "Method $methodName has return type with excessive generic nesting: $typeName. " +
                "Maximum supported generic depth is 3 levels."
            )
            return false
        }
        
        return true
    }

    /**
     * Validates that a parameter type is supported for code generation.
     */
    private fun validateParameterType(param: KSValueParameter, methodName: String): Boolean {
        val paramName = param.name?.asString() ?: "unnamed"
        val paramType = param.type.resolve()
        val typeName = paramType.declaration.qualifiedName?.asString() ?: paramType.toString()
        
        // Check for unsupported generic wildcards
        if (typeName.contains("*") && !typeName.contains("kotlin.Array")) {
            logger.error(
                "Method $methodName parameter '$paramName' has unsupported type with wildcards: $typeName. " +
                "Wildcard types are not supported in generated tracing wrappers."
            )
            return false
        }
        
        // Check for excessively nested generics
        val genericDepth = typeName.count { it == '<' }
        if (genericDepth > 3) {
            logger.error(
                "Method $methodName parameter '$paramName' has type with excessive generic nesting: $typeName. " +
                "Maximum supported generic depth is 3 levels."
            )
            return false
        }
        
        // Check for function types (not currently supported)
        if (typeName.contains("kotlin.Function") || typeName.contains("->")) {
            logger.error(
                "Method $methodName parameter '$paramName' has function type: $typeName. " +
                "Function types are not currently supported in tracing wrappers."
            )
            return false
        }
        
        return true
    }

    /**
     * Validates that the annotated symbol is an interface and meets requirements.
     * @deprecated Use validateAnnotatedSymbol instead for comprehensive validation
     */
    @Deprecated("Use validateAnnotatedSymbol for comprehensive validation")
    private fun validateInterface(symbol: KSClassDeclaration): Boolean {
        return validateAnnotatedSymbol(symbol)
    }

    /**
     * Generates the tracing wrapper implementation for the given interface.
     */
    private fun generateTracingWrapper(interfaceDeclaration: KSClassDeclaration) {
        logger.info("Generating tracing wrapper for ${interfaceDeclaration.simpleName.asString()}")
        
        val interfaceName = interfaceDeclaration.simpleName.asString()
        val packageName = interfaceDeclaration.packageName.asString()
        val generatedClassName = "Ksp$interfaceName"
        
        // Analyze methods using the method analyzer
        val methodAnalyzer = MethodAnalyzer()
        val methods = methodAnalyzer.analyzeInterface(interfaceDeclaration)
        
        logger.info("Analyzed ${methods.size} methods for $interfaceName")
        
        // Log method analysis results for verification
        methods.forEach { method ->
            logger.info("Method ${method.name}: ${method.parameters.size} params, returns ${method.returnType.type}")
            method.parameters.forEach { param ->
                logger.info("  - ${param.name}: ${param.type} (tensor: ${param.isTensor})")
            }
        }
        
        // Generate the tracing wrapper class
        val generator = TracingWrapperGenerator(codeGenerator, logger)
        generator.generateTracingWrapper(interfaceDeclaration, methods)
        
        logger.info("Code generation complete for $generatedClassName in package $packageName")
    }
}

/**
 * Provider for the TracingWrapperProcessor.
 * This class is responsible for creating instances of TracingWrapperProcessor
 * when the KSP framework needs to process annotations.
 */
class TracingWrapperProcessorProvider : SymbolProcessorProvider {
    override fun create(environment: SymbolProcessorEnvironment): SymbolProcessor {
        return TracingWrapperProcessor(environment.codeGenerator, environment.logger)
    }
}