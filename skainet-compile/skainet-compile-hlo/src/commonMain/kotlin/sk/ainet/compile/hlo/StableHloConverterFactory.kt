package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.ActivationOperationsConverter
import sk.ainet.compile.hlo.converters.ConstantOperationsConverter
import sk.ainet.compile.hlo.converters.GatherOperationsConverter
import sk.ainet.compile.hlo.converters.LegacyOperationsConverter
import sk.ainet.compile.hlo.converters.LinalgOperationsConverter
import sk.ainet.compile.hlo.converters.MathOperationsConverter
import sk.ainet.compile.hlo.converters.NeuralNetOperationsConverter
import sk.ainet.compile.hlo.converters.ReductionOperationsConverter
import sk.ainet.compile.hlo.converters.ShapeOperationsConverter
import kotlin.jvm.JvmStatic

/**
 * Factory for creating StableHLO converters with default configurations.
 * 
 * This factory provides convenient methods for creating converters with
 * commonly used operation converters and configurations.
 */
public object StableHloConverterFactory {
    
    /**
     * Create a converter with basic operations support (add, matmul, relu)
     */
    @JvmStatic
    public fun createBasic(): StableHloConverter {
        val registry = StableHloOperationRegistry()
        val typeMapper = TypeMapper()
        val validator = MlirValidator()
        
        // Register legacy operations for backward compatibility
        registry.register(LegacyOperationsConverter())
        
        // Register enhanced mathematical operations converter
        registry.register(MathOperationsConverter())
        
        // Register linear algebra operations converter
        registry.register(LinalgOperationsConverter())
        
        // Register activation operations converter
        registry.register(ActivationOperationsConverter())
        
        // Register shape operations converter
        registry.register(ShapeOperationsConverter())
        
        // Register reduction operations converter
        registry.register(ReductionOperationsConverter())

        // Register constant operations converter
        registry.register(ConstantOperationsConverter())

        // Register gather / embedding / index_select converter — the
        // LLM front-door op for token-id \u2192 embedding lookups.
        registry.register(GatherOperationsConverter())

        return StableHloConverter(registry, typeMapper, validator)
    }

    /**
     * Create a converter with extended operations support
     */
    @JvmStatic
    public fun createExtended(): StableHloConverter {
        val registry = StableHloOperationRegistry()
        val typeMapper = TypeMapper()
        val validator = MlirValidator()
        
        // Register legacy operations
        registry.register(LegacyOperationsConverter())
        
        // Register mathematical operations converter
        registry.register(MathOperationsConverter())
        
        // Register linear algebra operations converter
        registry.register(LinalgOperationsConverter())
        
        // Register neural network operations converter
        registry.register(NeuralNetOperationsConverter())
        
        // Register activation operations converter
        registry.register(ActivationOperationsConverter())
        
        // Register shape operations converter
        registry.register(ShapeOperationsConverter())

        // Register reduction operations converter
        registry.register(ReductionOperationsConverter())

        // Register constant operations converter
        registry.register(ConstantOperationsConverter())

        // Register gather / embedding / index_select converter — the
        // LLM front-door op for token-id \u2192 embedding lookups.
        registry.register(GatherOperationsConverter())

        return StableHloConverter(registry, typeMapper, validator)
    }

    /**
     * Create a converter without validation (for performance)
     */
    @JvmStatic
    public fun createFast(): StableHloConverter {
        val registry = StableHloOperationRegistry()
        val typeMapper = TypeMapper()
        
        registry.register(LegacyOperationsConverter())
        registry.register(MathOperationsConverter())
        registry.register(LinalgOperationsConverter())
        registry.register(NeuralNetOperationsConverter())
        registry.register(ActivationOperationsConverter())
        registry.register(ShapeOperationsConverter())
        registry.register(ReductionOperationsConverter())
        registry.register(ConstantOperationsConverter())

        return StableHloConverter(registry, typeMapper, null)
    }
    
    /**
     * Create a custom converter with the provided components
     */
    @JvmStatic
    @kotlin.jvm.JvmOverloads
    public fun createCustom(
        registry: StableHloOperationRegistry,
        typeMapper: TypeMapper = TypeMapper(),
        validator: MlirValidator? = MlirValidator()
    ): StableHloConverter {
        return StableHloConverter(registry, typeMapper, validator)
    }
}