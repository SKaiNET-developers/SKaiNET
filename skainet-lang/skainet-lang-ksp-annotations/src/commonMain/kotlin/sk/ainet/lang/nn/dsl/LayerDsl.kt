package sk.ainet.lang.nn.dsl

/**
 * Annotation to mark Module classes for automatic Network DSL generation.
 *
 * When applied to a Module class, the KSP processor will generate DSL methods
 * for NeuralNetworkDsl that allow easy creation of the layer within the DSL.
 *
 * Example:
 * ```kotlin
 * @LayerDsl
 * class Conv1d<T : DType, V>(
 *     val inChannels: Int,
 *     val outChannels: Int,
 *     val kernelSize: Int,
 *     ...
 * ) : Module<T, V>()
 * ```
 *
 * This will generate DSL methods in NeuralNetworkDsl interface and implementations
 * that allow building Conv1d layers within the sequential {} DSL.
 *
 * @param dslName Optional custom name for the DSL method (defaults to camelCase of class name)
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
public annotation class LayerDsl(
    val dslName: String = ""
)
