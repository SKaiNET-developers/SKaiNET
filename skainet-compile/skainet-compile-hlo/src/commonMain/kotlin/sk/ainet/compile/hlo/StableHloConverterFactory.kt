package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.ActivationOperationsConverter
import sk.ainet.compile.hlo.converters.ArgMaxOperationsConverter
import sk.ainet.compile.hlo.converters.AttentionOperationsConverter
import sk.ainet.compile.hlo.converters.ConstantOperationsConverter
import sk.ainet.compile.hlo.converters.GatherOperationsConverter
import sk.ainet.compile.hlo.converters.LegacyOperationsConverter
import sk.ainet.compile.hlo.converters.LinalgOperationsConverter
import sk.ainet.compile.hlo.converters.MathOperationsConverter
import sk.ainet.compile.hlo.converters.NeuralNetOperationsConverter
import sk.ainet.compile.hlo.converters.ReductionOperationsConverter
import sk.ainet.compile.hlo.converters.ScalarOperationsConverter
import sk.ainet.compile.hlo.converters.ShapeOperationsConverter
import sk.ainet.compile.hlo.converters.UnaryMathConverter
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
     *
     * @param policy Controls inline vs external constant materialization.
     *     Defaults to [ConstantMaterializationPolicy.InlineAlways] for
     *     backward compatibility; see issue #523.
     */
    @JvmStatic
    @kotlin.jvm.JvmOverloads
    public fun createBasic(
        policy: ConstantMaterializationPolicy = ConstantMaterializationPolicy.InlineAlways,
        target: String? = null,
        granularity: sk.ainet.compile.target.OpGranularityPolicy? = null,
        errorPolicy: ConversionErrorPolicy = ConversionErrorPolicy.STRICT
    ): StableHloConverter {
        val registry = StableHloOperationRegistry()
        val typeMapper = TypeMapper()
        val validator = MlirValidator()
        
        // Register legacy operations for backward compatibility
        registry.register(LegacyOperationsConverter())
        
        // Register enhanced mathematical operations converter
        registry.register(MathOperationsConverter())
        
        // Register linear algebra operations converter
        registry.register(LinalgOperationsConverter())

        // Register neural network operations converter (conv / pool / norms).
        // Registered in the same relative position as in createExtended so
        // op-name precedence (last-writer-wins per name) stays identical —
        // e.g. AttentionOperationsConverter still wins
        // scaledDotProductAttention. Previously missing here, so a traced
        // model with norms could not lower via createBasic (#1247).
        registry.register(NeuralNetOperationsConverter())

        // Register activation operations converter
        registry.register(ActivationOperationsConverter())

        // Register shape operations converter
        registry.register(ShapeOperationsConverter())

        // Register reduction operations converter
        registry.register(ReductionOperationsConverter())
        // argMax: logits -> index, lowered to reduce-max + broadcast + compare + iota + select + reduce-min
        registry.register(ArgMaxOperationsConverter())

        // Register elementwise unary math converter (sqrt, exp, log, abs, …).
        // Must be present so downstream consumers don't cascade-fail with
        // "wrong arity" when an upstream op is silently dropped.
        registry.register(UnaryMathConverter())

        // Register tensor+scalar ops (addScalar / mulScalar / …) emitted by the
        // KSP-generated tracing wrapper for `tensor op Number` expressions.
        registry.register(ScalarOperationsConverter())

        // Register constant operations converter
        registry.register(ConstantOperationsConverter())

        // Register attention (scaledDotProductAttention) converter
        registry.register(AttentionOperationsConverter())

        // Register gather / embedding / index_select converter — the
        // LLM front-door op for token-id \u2192 embedding lookups.
        registry.register(GatherOperationsConverter())

        return StableHloConverter(registry, typeMapper, validator, policy, target, granularity, errorPolicy)
    }

    /**
     * Create a converter with extended operations support
     *
     * @param policy See [createBasic].
     */
    @JvmStatic
    @kotlin.jvm.JvmOverloads
    public fun createExtended(
        policy: ConstantMaterializationPolicy = ConstantMaterializationPolicy.InlineAlways,
        errorPolicy: ConversionErrorPolicy = ConversionErrorPolicy.STRICT
    ): StableHloConverter {
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
        // argMax: logits -> index, lowered to reduce-max + broadcast + compare + iota + select + reduce-min
        registry.register(ArgMaxOperationsConverter())

        // Register elementwise unary math converter (sqrt, exp, log, abs, …).
        // Must be present so downstream consumers don't cascade-fail with
        // "wrong arity" when an upstream op is silently dropped.
        registry.register(UnaryMathConverter())

        // Register tensor+scalar ops (addScalar / mulScalar / …) emitted by the
        // KSP-generated tracing wrapper for `tensor op Number` expressions.
        registry.register(ScalarOperationsConverter())

        // Register constant operations converter
        registry.register(ConstantOperationsConverter())

        // Register attention (scaledDotProductAttention) converter
        registry.register(AttentionOperationsConverter())

        // Register gather / embedding / index_select converter — the
        // LLM front-door op for token-id \u2192 embedding lookups.
        registry.register(GatherOperationsConverter())

        return StableHloConverter(registry, typeMapper, validator, policy, errorPolicy = errorPolicy)
    }

    /**
     * Create a converter without validation (for performance)
     *
     * @param policy See [createBasic].
     */
    @JvmStatic
    @kotlin.jvm.JvmOverloads
    public fun createFast(
        policy: ConstantMaterializationPolicy = ConstantMaterializationPolicy.InlineAlways,
        errorPolicy: ConversionErrorPolicy = ConversionErrorPolicy.STRICT
    ): StableHloConverter {
        val registry = StableHloOperationRegistry()
        val typeMapper = TypeMapper()

        registry.register(LegacyOperationsConverter())
        registry.register(MathOperationsConverter())
        registry.register(LinalgOperationsConverter())
        registry.register(NeuralNetOperationsConverter())
        registry.register(ActivationOperationsConverter())
        registry.register(ShapeOperationsConverter())
        registry.register(ReductionOperationsConverter())
        // argMax: logits -> index, lowered to reduce-max + broadcast + compare + iota + select + reduce-min
        registry.register(ArgMaxOperationsConverter())
        registry.register(UnaryMathConverter())
        registry.register(ScalarOperationsConverter())
        registry.register(ConstantOperationsConverter())

        return StableHloConverter(registry, typeMapper, null, policy, errorPolicy = errorPolicy)
    }

    /**
     * Create a custom converter with the provided components
     *
     * @param policy See [createBasic].
     */
    @JvmStatic
    @kotlin.jvm.JvmOverloads
    public fun createCustom(
        registry: StableHloOperationRegistry,
        typeMapper: TypeMapper = TypeMapper(),
        validator: MlirValidator? = MlirValidator(),
        policy: ConstantMaterializationPolicy = ConstantMaterializationPolicy.InlineAlways,
        errorPolicy: ConversionErrorPolicy = ConversionErrorPolicy.STRICT
    ): StableHloConverter {
        return StableHloConverter(registry, typeMapper, validator, policy, errorPolicy = errorPolicy)
    }
}