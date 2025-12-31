package sk.ainet.compile.hlo.validation

/**
 * Parsed structure of an MLIR module
 */
public data class ParsedMlirStructure(
    val moduleName: String? = null,
    val functionName: String,
    val functionSignature: FunctionSignature,
    val operations: List<ParsedOperation>,
    val metadata: Map<String, String> = emptyMap()
) {
    /**
     * Convert back to MLIR string representation
     */
    public fun toMlirString(): String {
        val sb = StringBuilder()
        
        sb.appendLine("module {")
        sb.appendLine("  func.func @${functionName}${functionSignature.toMlirString()} {")
        
        operations.forEach { op ->
            sb.appendLine("    ${op.toMlirString()}")
        }
        
        sb.appendLine("    return")
        sb.appendLine("  }")
        sb.appendLine("}")
        
        return sb.toString()
    }
}

/**
 * Represents a function signature in MLIR
 */
public data class FunctionSignature(
    val parameters: List<Parameter>,
    val returnTypes: List<String>
) {
    public fun toMlirString(): String {
        val params = parameters.joinToString(", ") { "${it.name}: ${it.type}" }
        val returns = if (returnTypes.isEmpty()) "()" else "(${returnTypes.joinToString(", ")})"
        return "($params) -> $returns"
    }
}

/**
 * Represents a function parameter
 */
public data class Parameter(
    val name: String,
    val type: String
)

/**
 * Represents a parsed MLIR operation
 */
public data class ParsedOperation(
    val resultName: String?,
    val operationType: String,
    val operands: List<String>,
    val attributes: Map<String, String> = emptyMap(),
    val resultType: String? = null
) {
    public fun toMlirString(): String {
        val result = if (resultName != null) "$resultName = " else ""
        val operandStr = operands.joinToString(", ")
        val attrStr = if (attributes.isNotEmpty()) {
            " {" + attributes.entries.joinToString(", ") { "${it.key} = ${it.value}" } + "}"
        } else ""
        val typeStr = if (resultType != null) " : $resultType" else ""
        
        return "$result$operationType $operandStr$attrStr$typeStr"
    }
}

/**
 * Parser for MLIR StableHLO content.
 * 
 * This parser provides basic parsing capabilities for StableHLO MLIR modules,
 * extracting structure and operations for validation purposes.
 */
public class MlirParser {
    
    /**
     * Parse MLIR content into structured representation
     */
    public fun parse(content: String): Result<ParsedMlirStructure> {
        return try {
            val lines = content.lines().map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("//") }
            
            if (lines.isEmpty()) {
                return Result.failure(IllegalArgumentException("Empty MLIR content"))
            }
            
            val moduleMatch = findModuleDeclaration(lines)
            val functionMatch = findFunctionDeclaration(lines)
            
            if (functionMatch == null) {
                return Result.failure(IllegalArgumentException("No function declaration found"))
            }
            
            val functionName = extractFunctionName(functionMatch)
            val functionSignature = extractFunctionSignature(functionMatch)
            val operations = extractOperations(lines)
            
            val structure = ParsedMlirStructure(
                functionName = functionName,
                functionSignature = functionSignature,
                operations = operations
            )
            
            Result.success(structure)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Validate that content can be parsed without errors
     */
    public fun validateParsability(content: String): List<String> {
        val errors = mutableListOf<String>()
        
        try {
            val parseResult = parse(content)
            if (parseResult.isFailure) {
                errors.add("Parse failure: ${parseResult.exceptionOrNull()?.message}")
            }
        } catch (e: Exception) {
            errors.add("Parsing exception: ${e.message}")
        }
        
        return errors
    }
    
    private fun findModuleDeclaration(lines: List<String>): String? {
        return lines.find { it.startsWith("module") }
    }
    
    private fun findFunctionDeclaration(lines: List<String>): String? {
        return lines.find { it.contains("func.func") }
    }
    
    private fun extractFunctionName(functionLine: String): String {
        val regex = Regex("""func\.func @(\w+)""")
        return regex.find(functionLine)?.groupValues?.get(1) ?: "unknown"
    }
    
    private fun extractFunctionSignature(functionLine: String): FunctionSignature {
        val parameters = extractParameters(functionLine)
        val returnTypes = extractReturnTypes(functionLine)
        
        return FunctionSignature(parameters, returnTypes)
    }
    
    private fun extractParameters(functionLine: String): List<Parameter> {
        val parameters = mutableListOf<Parameter>()
        
        // Extract parameter section: @name(param1: type1, param2: type2)
        val paramRegex = Regex("""\(([^)]*)\)""")
        val paramMatch = paramRegex.find(functionLine)
        
        if (paramMatch != null && paramMatch.groupValues[1].isNotEmpty()) {
            val paramString = paramMatch.groupValues[1]
            val paramPairs = paramString.split(",").map { it.trim() }
            
            for (paramPair in paramPairs) {
                if (paramPair.contains(":")) {
                    val parts = paramPair.split(":", limit = 2)
                    if (parts.size == 2) {
                        parameters.add(Parameter(parts[0].trim(), parts[1].trim()))
                    }
                }
            }
        }
        
        return parameters
    }
    
    private fun extractReturnTypes(functionLine: String): List<String> {
        val returnTypes = mutableListOf<String>()
        
        // Extract return type section: -> (type1, type2) or -> ()
        val returnRegex = Regex("""->\s*\(([^)]*)\)""")
        val returnMatch = returnRegex.find(functionLine)
        
        if (returnMatch != null && returnMatch.groupValues[1].isNotEmpty()) {
            val returnString = returnMatch.groupValues[1]
            returnTypes.addAll(returnString.split(",").map { it.trim() })
        }
        
        return returnTypes
    }
    
    private fun extractOperations(lines: List<String>): List<ParsedOperation> {
        val operations = mutableListOf<ParsedOperation>()
        
        for (line in lines) {
            if (line.contains("stablehlo.") || line.contains(" = ")) {
                val operation = parseOperation(line)
                if (operation != null) {
                    operations.add(operation)
                }
            }
        }
        
        return operations
    }
    
    private fun parseOperation(line: String): ParsedOperation? {
        try {
            // Handle assignment operations: %result = operation operands : type
            if (line.contains(" = ")) {
                val parts = line.split(" = ", limit = 2)
                if (parts.size == 2) {
                    val resultName = parts[0].trim()
                    val operationPart = parts[1].trim()
                    
                    return parseOperationPart(operationPart, resultName)
                }
            } else if (line.contains("stablehlo.")) {
                // Handle operations without assignment
                return parseOperationPart(line, null)
            }
            
            return null
        } catch (e: Exception) {
            // Skip malformed operations
            return null
        }
    }
    
    private fun parseOperationPart(operationPart: String, resultName: String?): ParsedOperation? {
        // Extract operation type
        val opTypeRegex = Regex("""(stablehlo\.\w+)""")
        val opTypeMatch = opTypeRegex.find(operationPart)
        
        if (opTypeMatch == null) {
            return null
        }
        
        val operationType = opTypeMatch.value
        
        // Extract operands (simplified - assumes space-separated operands before :)
        val operands = extractOperands(operationPart)
        
        // Extract result type (after :)
        val resultType = extractResultType(operationPart)
        
        return ParsedOperation(
            resultName = resultName,
            operationType = operationType,
            operands = operands,
            resultType = resultType
        )
    }
    
    private fun extractOperands(operationPart: String): List<String> {
        val operands = mutableListOf<String>()
        
        // Find operands between operation name and type annotation
        val beforeType = if (operationPart.contains(" : ")) {
            operationPart.substringBefore(" : ")
        } else {
            operationPart
        }
        
        // Remove operation name
        val afterOpName = if (beforeType.contains("stablehlo.")) {
            val opNameEnd = beforeType.indexOf("stablehlo.") + beforeType.substring(beforeType.indexOf("stablehlo.")).indexOf(' ')
            if (opNameEnd > 0 && opNameEnd < beforeType.length) {
                beforeType.substring(opNameEnd).trim()
            } else {
                ""
            }
        } else {
            beforeType
        }
        
        // Extract SSA values (starting with %)
        val ssaRegex = Regex("""%[a-zA-Z0-9_]+""")
        ssaRegex.findAll(afterOpName).forEach { match ->
            operands.add(match.value)
        }
        
        return operands
    }
    
    private fun extractResultType(operationPart: String): String? {
        return if (operationPart.contains(" : ")) {
            operationPart.substringAfter(" : ").trim()
        } else {
            null
        }
    }
}