package sk.ainet.compile.hlo.validation

import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.compile.hlo.MlirValidator
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Result of round-trip validation
 */
public sealed class RoundTripValidationResult {
    public data class Success(
        val originalModule: StableHloModule,
        val parsedModule: StableHloModule,
        val equivalenceReport: SemanticEquivalenceReport
    ) : RoundTripValidationResult()
    
    public data class Failure(
        val errors: List<String>,
        val stage: ValidationStage
    ) : RoundTripValidationResult()
}

/**
 * Stages of validation where failures can occur
 */
public enum class ValidationStage {
    SYNTAX_VALIDATION,
    PARSING,
    SEMANTIC_ANALYSIS,
    EQUIVALENCE_CHECK
}

/**
 * Report on semantic equivalence between original and parsed modules
 */
public data class SemanticEquivalenceReport(
    val isEquivalent: Boolean,
    val functionSignatureMatch: Boolean,
    val operationCountMatch: Boolean,
    val ssaStructureMatch: Boolean,
    val typeConsistency: Boolean,
    val differences: List<String> = emptyList()
)

/**
 * Validator for round-trip validation of StableHLO modules.
 * 
 * This class provides capabilities to validate that generated MLIR can be parsed
 * and maintains semantic equivalence with the original computation graph.
 */
public class RoundTripValidator(
    private val mlirValidator: MlirValidator = MlirValidator(),
    private val parser: MlirParser = MlirParser()
) {
    
    /**
     * Perform complete round-trip validation of a StableHLO module
     */
    public fun validateRoundTrip(module: StableHloModule): RoundTripValidationResult {
        // Step 1: Validate syntax of original module
        val syntaxErrors = mlirValidator.validate(module.content)
        if (syntaxErrors.isNotEmpty()) {
            return RoundTripValidationResult.Failure(
                errors = syntaxErrors,
                stage = ValidationStage.SYNTAX_VALIDATION
            )
        }
        
        // Step 2: Parse the MLIR content
        val parseResult = parser.parse(module.content)
        if (parseResult.isFailure) {
            return RoundTripValidationResult.Failure(
                errors = listOf("Failed to parse MLIR: ${parseResult.exceptionOrNull()?.message}"),
                stage = ValidationStage.PARSING
            )
        }
        
        val parsedStructure = parseResult.getOrThrow()
        
        // Step 3: Reconstruct module from parsed structure
        val reconstructedModule = reconstructModule(parsedStructure, module)
        
        // Step 4: Perform semantic equivalence check
        val equivalenceReport = checkSemanticEquivalence(module, reconstructedModule)
        
        return if (equivalenceReport.isEquivalent) {
            RoundTripValidationResult.Success(
                originalModule = module,
                parsedModule = reconstructedModule,
                equivalenceReport = equivalenceReport
            )
        } else {
            RoundTripValidationResult.Failure(
                errors = equivalenceReport.differences,
                stage = ValidationStage.EQUIVALENCE_CHECK
            )
        }
    }
    
    /**
     * Check semantic equivalence between two StableHLO modules
     */
    public fun checkSemanticEquivalence(
        original: StableHloModule,
        parsed: StableHloModule
    ): SemanticEquivalenceReport {
        val differences = mutableListOf<String>()
        
        // Check function signature equivalence
        val functionSignatureMatch = checkFunctionSignatures(original, parsed, differences)
        
        // Check operation count and types
        val operationCountMatch = checkOperationCounts(original, parsed, differences)
        
        // Check SSA structure
        val ssaStructureMatch = checkSSAStructure(original, parsed, differences)
        
        // Check type consistency
        val typeConsistency = checkTypeConsistency(original, parsed, differences)
        
        val isEquivalent = functionSignatureMatch && operationCountMatch && 
                          ssaStructureMatch && typeConsistency
        
        return SemanticEquivalenceReport(
            isEquivalent = isEquivalent,
            functionSignatureMatch = functionSignatureMatch,
            operationCountMatch = operationCountMatch,
            ssaStructureMatch = ssaStructureMatch,
            typeConsistency = typeConsistency,
            differences = differences
        )
    }
    
    /**
     * Validate that a module can be successfully parsed and reconstructed
     */
    public fun validateParsability(module: StableHloModule): List<String> {
        val errors = mutableListOf<String>()
        
        // Basic syntax validation
        errors.addAll(mlirValidator.validate(module.content))
        
        // Parse validation
        val parseResult = parser.parse(module.content)
        if (parseResult.isFailure) {
            errors.add("Parse failure: ${parseResult.exceptionOrNull()?.message}")
        }
        
        return errors
    }
    
    private fun reconstructModule(
        parsedStructure: ParsedMlirStructure,
        originalModule: StableHloModule
    ): StableHloModule {
        // Reconstruct the module content from parsed structure
        val reconstructedContent = parsedStructure.toMlirString()
        
        return StableHloModule(
            content = reconstructedContent,
            functionName = parsedStructure.functionName,
            inputSpecs = originalModule.inputSpecs,
            outputSpecs = originalModule.outputSpecs,
            metadata = originalModule.metadata + mapOf("reconstructed" to true)
        )
    }
    
    private fun checkFunctionSignatures(
        original: StableHloModule,
        parsed: StableHloModule,
        differences: MutableList<String>
    ): Boolean {
        val originalSig = extractFunctionSignature(original.content)
        val parsedSig = extractFunctionSignature(parsed.content)
        
        if (originalSig != parsedSig) {
            differences.add("Function signature mismatch: '$originalSig' vs '$parsedSig'")
            return false
        }
        return true
    }
    
    private fun checkOperationCounts(
        original: StableHloModule,
        parsed: StableHloModule,
        differences: MutableList<String>
    ): Boolean {
        val originalOps = extractOperations(original.content)
        val parsedOps = extractOperations(parsed.content)
        
        if (originalOps.size != parsedOps.size) {
            differences.add("Operation count mismatch: ${originalOps.size} vs ${parsedOps.size}")
            return false
        }
        
        // Check operation types match
        val originalOpTypes = originalOps.map { it.type }.sorted()
        val parsedOpTypes = parsedOps.map { it.type }.sorted()
        
        if (originalOpTypes != parsedOpTypes) {
            differences.add("Operation types mismatch: $originalOpTypes vs $parsedOpTypes")
            return false
        }
        
        return true
    }
    
    private fun checkSSAStructure(
        original: StableHloModule,
        parsed: StableHloModule,
        differences: MutableList<String>
    ): Boolean {
        val originalSSA = extractSSAStructure(original.content)
        val parsedSSA = extractSSAStructure(parsed.content)
        
        if (originalSSA.definedValues.size != parsedSSA.definedValues.size) {
            differences.add("SSA value count mismatch: ${originalSSA.definedValues.size} vs ${parsedSSA.definedValues.size}")
            return false
        }
        
        return true
    }
    
    private fun checkTypeConsistency(
        original: StableHloModule,
        parsed: StableHloModule,
        differences: MutableList<String>
    ): Boolean {
        val originalTypes = extractTypes(original.content)
        val parsedTypes = extractTypes(parsed.content)
        
        if (originalTypes != parsedTypes) {
            differences.add("Type consistency mismatch")
            return false
        }
        
        return true
    }
    
    private fun extractFunctionSignature(content: String): String {
        val funcRegex = Regex("""func\.func @(\w+)\([^)]*\)\s*->\s*\([^)]*\)""")
        return funcRegex.find(content)?.value ?: ""
    }
    
    private fun extractOperations(content: String): List<MlirOperation> {
        val operations = mutableListOf<MlirOperation>()
        val lines = content.lines()
        
        for (line in lines) {
            val trimmed = line.trim()
            if (trimmed.contains("stablehlo.")) {
                val opType = extractOperationType(trimmed)
                if (opType.isNotEmpty()) {
                    operations.add(MlirOperation(type = opType, line = trimmed))
                }
            }
        }
        
        return operations
    }
    
    private fun extractOperationType(line: String): String {
        val regex = Regex("""stablehlo\.(\w+)""")
        return regex.find(line)?.groupValues?.get(1) ?: ""
    }
    
    private fun extractSSAStructure(content: String): SSAStructure {
        val definedValues = mutableSetOf<String>()
        val usedValues = mutableSetOf<String>()
        val lines = content.lines()
        
        for (line in lines) {
            val trimmed = line.trim()
            
            // Extract defined values
            if (trimmed.contains(" = ")) {
                val parts = trimmed.split(" = ", limit = 2)
                if (parts.size == 2) {
                    val valueName = parts[0].trim()
                    if (valueName.startsWith("%")) {
                        definedValues.add(valueName)
                    }
                }
            }
            
            // Extract used values
            val regex = Regex("""%[a-zA-Z0-9_]+""")
            regex.findAll(trimmed).forEach { match ->
                usedValues.add(match.value)
            }
        }
        
        return SSAStructure(definedValues, usedValues)
    }
    
    private fun extractTypes(content: String): Set<String> {
        val types = mutableSetOf<String>()
        val typeRegex = Regex("""tensor<[^>]+>""")
        
        typeRegex.findAll(content).forEach { match ->
            types.add(match.value)
        }
        
        return types
    }
}

/**
 * Represents an MLIR operation for comparison
 */
private data class MlirOperation(
    val type: String,
    val line: String
)

/**
 * Represents SSA structure for comparison
 */
private data class SSAStructure(
    val definedValues: Set<String>,
    val usedValues: Set<String>
)