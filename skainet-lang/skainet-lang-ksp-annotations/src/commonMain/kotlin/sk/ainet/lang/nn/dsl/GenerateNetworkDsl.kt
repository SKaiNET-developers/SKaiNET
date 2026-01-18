package sk.ainet.lang.nn.dsl

/**
 * Annotation to mark an interface for automatic Network DSL extension generation via KSP.
 *
 * When applied to an interface (like TensorOps), the KSP processor will scan for
 * methods annotated with @ActivationDsl and generate corresponding extension functions
 * for NeuralNetworkDsl.
 *
 * The generated file will contain extension functions that integrate activation
 * methods into the neural network DSL builder pattern.
 *
 * ## Usage
 *
 * ```kotlin
 * @GenerateNetworkDsl
 * interface TensorOps {
 *     @ActivationDsl
 *     fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V>
 *
 *     @ActivationDsl
 *     fun <T : DType, V> leakyRelu(tensor: Tensor<T, V>, negativeSlope: Float = 0.01f): Tensor<T, V>
 * }
 * ```
 *
 * This will generate:
 *
 * ```kotlin
 * // In TensorOpsNetworkDsl.kt
 * @NetworkDsl
 * fun <T : DType, V> NeuralNetworkDsl<T, V>.relu(id: String = "") {
 *     activation(id) { it.relu() }
 * }
 *
 * @NetworkDsl
 * fun <T : DType, V> NeuralNetworkDsl<T, V>.leakyRelu(negativeSlope: Float = 0.01f, id: String = "") {
 *     activation(id) { it.leakyRelu(negativeSlope) }
 * }
 * ```
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
public annotation class GenerateNetworkDsl
