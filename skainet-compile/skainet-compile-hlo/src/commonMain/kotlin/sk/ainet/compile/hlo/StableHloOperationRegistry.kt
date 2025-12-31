package sk.ainet.compile.hlo

import sk.ainet.lang.graph.GraphNode

/**
 * Interface for converting specific operations to StableHLO format.
 * 
 * Each converter is responsible for handling one or more operation types
 * and generating the appropriate StableHLO MLIR code.
 */
public interface StableHloOperationConverter {
    /**
     * Set of operation names this converter supports
     */
    public val supportedOperations: Set<String>
    
    /**
     * Convert a graph node to StableHLO operations
     * 
     * @param node The graph node to convert
     * @param operands List of SSA value names for the operands
     * @param context Conversion context for state management
     * @return Result of the conversion
     */
    public fun convert(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult
}

/**
 * Result of an operation conversion
 */
public sealed class ConversionResult {
    /**
     * Successful conversion
     * @param outputValueName The SSA value name of the result
     * @param emittedOperations List of MLIR operations that were emitted
     */
    public data class Success(
        val outputValueName: String,
        val emittedOperations: List<String> = emptyList()
    ) : ConversionResult()
    
    /**
     * Conversion failed with an error
     * @param error Description of the error
     * @param fallbackComment Optional comment to emit as fallback
     */
    public data class Failure(
        val error: String,
        val fallbackComment: String? = null
    ) : ConversionResult()
    
    /**
     * Operation is not supported by this converter
     * @param operationName Name of the unsupported operation
     * @param reason Reason why it's not supported
     */
    public data class Unsupported(
        val operationName: String,
        val reason: String
    ) : ConversionResult()
}

/**
 * Registry for StableHLO operation converters.
 * 
 * This class manages the registration and lookup of operation converters,
 * providing a pluggable system for extending StableHLO support.
 */
public class StableHloOperationRegistry {
    private val converters = mutableMapOf<String, StableHloOperationConverter>()
    
    /**
     * Register a converter for one or more operations
     */
    public fun register(converter: StableHloOperationConverter) {
        for (operation in converter.supportedOperations) {
            converters[operation] = converter
        }
    }
    
    /**
     * Get the converter for a specific operation name
     */
    public fun getConverter(operationName: String): StableHloOperationConverter? {
        return converters[operationName]
    }
    
    /**
     * Get all supported operation names
     */
    public fun getSupportedOperations(): Set<String> {
        return converters.keys.toSet()
    }
    
    /**
     * Check if an operation is supported
     */
    public fun isSupported(operationName: String): Boolean {
        return converters.containsKey(operationName)
    }
    
    /**
     * Unregister a converter for specific operations
     */
    public fun unregister(operationNames: Set<String>) {
        for (operation in operationNames) {
            converters.remove(operation)
        }
    }
    
    /**
     * Clear all registered converters
     */
    public fun clear() {
        converters.clear()
    }
    
    /**
     * Get statistics about registered converters
     */
    public fun getStats(): RegistryStats {
        val converterCount = converters.values.toSet().size
        val operationCount = converters.size
        return RegistryStats(converterCount, operationCount)
    }
}

/**
 * Statistics about the operation registry
 */
public data class RegistryStats(
    val converterCount: Int,
    val operationCount: Int
)