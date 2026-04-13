package sk.ainet.compile.hlo

/**
 * Validator for MLIR syntax and semantic correctness.
 * 
 * This class provides validation capabilities for generated StableHLO MLIR code,
 * checking for syntax errors, semantic issues, and best practices.
 */
public class MlirValidator {
    
    /**
     * Validate MLIR content and return list of errors (empty if valid)
     */
    public fun validate(content: String): List<String> {
        val errors = mutableListOf<String>()
        
        // Basic syntax validation
        errors.addAll(validateSyntax(content))
        
        // Semantic validation
        errors.addAll(validateSemantics(content))
        
        return errors
    }
    
    /**
     * Validate basic MLIR syntax
     */
    private fun validateSyntax(content: String): List<String> {
        val errors = mutableListOf<String>()
        val lines = content.lines()
        
        var braceCount = 0
        var inModule = false
        var inFunction = false
        
        for ((lineNum, line) in lines.withIndex()) {
            val trimmed = line.trim()

            // Skip empty lines and comments
            if (trimmed.isEmpty() || trimmed.startsWith("//")) continue

            // Check brace balance
            braceCount += trimmed.count { it == '{' }
            braceCount -= trimmed.count { it == '}' }

            // Check module structure
            if (trimmed.startsWith("module")) {
                if (inModule) {
                    errors.add("Line ${lineNum + 1}: Nested modules not allowed")
                }
                inModule = true
                // Module headers may carry a `module attributes { ... } {`
                // preamble whose attribute dict contains `name = "value"`
                // entries. These aren't SSA assignments and must not be
                // fed into validateSSAValue, so stop processing this line
                // here.
                continue
            }

            // Check function structure
            if (trimmed.contains("func.func")) {
                if (!inModule) {
                    errors.add("Line ${lineNum + 1}: Function must be inside module")
                }
                inFunction = true
            }

            // Check for basic SSA value format
            if (trimmed.contains(" = ") && !validateSSAValue(trimmed)) {
                errors.add("Line ${lineNum + 1}: Invalid SSA value format")
            }
            
            // Check for proper operation format
            if (trimmed.contains("stablehlo.") && !validateStableHloOperation(trimmed)) {
                errors.add("Line ${lineNum + 1}: Invalid StableHLO operation format")
            }
        }
        
        // Check final brace balance
        if (braceCount != 0) {
            errors.add("Unbalanced braces: $braceCount")
        }
        
        return errors
    }
    
    /**
     * Validate semantic correctness
     */
    private fun validateSemantics(content: String): List<String> {
        val errors = mutableListOf<String>()
        val lines = content.lines()
        val definedValues = mutableSetOf<String>()
        val usedValues = mutableSetOf<String>()
        
        for ((lineNum, line) in lines.withIndex()) {
            val trimmed = line.trim()

            // Skip empty lines, comments, and module header lines (which
            // may carry a `module attributes { ... }` dictionary whose
            // `name = "value"` entries look like SSA assignments but are
            // not).
            if (trimmed.isEmpty() || trimmed.startsWith("//") || trimmed.startsWith("module")) continue

            // Extract defined SSA values
            if (trimmed.contains(" = ")) {
                val parts = trimmed.split(" = ", limit = 2)
                if (parts.size == 2) {
                    val valueName = parts[0].trim()
                    if (valueName.startsWith("%")) {
                        if (definedValues.contains(valueName)) {
                            errors.add("Line ${lineNum + 1}: SSA value $valueName redefined")
                        }
                        definedValues.add(valueName)
                    }
                }
            }
            
            // Extract used SSA values
            val usedInLine = extractUsedValues(trimmed)
            usedValues.addAll(usedInLine)
        }
        
        // Check for undefined values (excluding function arguments)
        for (used in usedValues) {
            if (!used.startsWith("%arg") && !definedValues.contains(used)) {
                errors.add("Undefined SSA value: $used")
            }
        }
        
        return errors
    }
    
    /**
     * Validate SSA value format
     */
    private fun validateSSAValue(line: String): Boolean {
        val parts = line.split(" = ", limit = 2)
        if (parts.size != 2) return false
        
        val valueName = parts[0].trim()
        return valueName.startsWith("%") && valueName.length > 1
    }
    
    /**
     * Validate StableHLO operation format
     */
    private fun validateStableHloOperation(line: String): Boolean {
        // Basic check for StableHLO operation format
        return line.contains("stablehlo.") && 
               (line.contains(" : ") || line.contains("->"))
    }
    
    /**
     * Extract SSA values used in a line
     */
    private fun extractUsedValues(line: String): Set<String> {
        val values = mutableSetOf<String>()
        val regex = Regex("%[a-zA-Z0-9_]+")
        
        regex.findAll(line).forEach { match ->
            values.add(match.value)
        }
        
        return values
    }
    
    /**
     * Validate that the content represents a complete MLIR module
     */
    public fun validateModule(content: String): List<String> {
        val errors = mutableListOf<String>()

        // Accept both the bare `module {` and the attributes-carrying
        // `module attributes { ... } {` header forms.
        if (!content.contains("module {") && !content.contains("module attributes")) {
            errors.add("Missing module declaration")
        }
        
        if (!content.contains("func.func")) {
            errors.add("Missing function declaration")
        }
        
        if (!content.contains("return")) {
            errors.add("Missing return statement")
        }
        
        return errors + validate(content)
    }
}