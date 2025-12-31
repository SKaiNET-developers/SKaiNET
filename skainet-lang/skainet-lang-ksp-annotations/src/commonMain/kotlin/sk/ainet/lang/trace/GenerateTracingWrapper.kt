package sk.ainet.lang.trace

/**
 * Annotation to mark interfaces for automatic tracing wrapper generation via KSP.
 * 
 * When applied to an interface, the KSP processor will generate a tracing wrapper class
 * that implements the interface and delegates all method calls to a base implementation
 * while emitting OpTrace events for computation graph recording.
 * 
 * The generated class will be named "Ksp" + interface name (e.g., KspTensorOps)
 * and will be placed in the same package as the annotated interface.
 * 
 * ## Usage
 * 
 * ```kotlin
 * @GenerateTracingWrapper
 * interface TensorOps {
 *     fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
 *     // ... other methods
 * }
 * ```
 * 
 * This will generate:
 * 
 * ```kotlin
 * class KspTensorOps(
 *     private val base: TensorOps,
 *     private val sink: OpSink,
 *     private val session: TraceSession = TraceSession()
 * ) : TensorOps {
 *     override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
 *         val out = base.add(a, b)
 *         // ... tracing logic
 *         return out
 *     }
 *     // ... all other methods with tracing
 * }
 * ```
 * 
 * ## Requirements
 * 
 * - Can only be applied to interfaces
 * - The interface must be processable by KSP (no complex generic constraints)
 * - All method parameters and return types must be supported by the code generator
 * 
 * @see sk.ainet.lang.trace.OpTrace
 * @see sk.ainet.lang.trace.OpSink
 * @see sk.ainet.lang.trace.TraceSession
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
public annotation class GenerateTracingWrapper