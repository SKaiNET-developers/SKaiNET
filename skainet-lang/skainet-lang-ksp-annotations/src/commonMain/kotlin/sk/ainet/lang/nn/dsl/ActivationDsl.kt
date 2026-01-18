package sk.ainet.lang.nn.dsl

/**
 * Annotation to mark TensorOps methods for automatic Network DSL generation.
 *
 * When applied to activation methods in TensorOps, the KSP processor will generate
 * extension functions for NeuralNetworkDsl that provide convenient DSL methods.
 *
 * Example:
 * ```kotlin
 * @ActivationDsl
 * fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V>
 * ```
 *
 * This will generate:
 * ```kotlin
 * fun <T : DType, V> NeuralNetworkDsl<T, V>.relu(id: String = "") {
 *     activation(id) { it.relu() }
 * }
 * ```
 *
 * @param dslName Optional custom name for the DSL method (defaults to the method name)
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.SOURCE)
public annotation class ActivationDsl(
    val dslName: String = ""
)
