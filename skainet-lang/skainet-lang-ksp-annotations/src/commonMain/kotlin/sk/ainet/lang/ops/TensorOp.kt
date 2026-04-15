package  sk.ainet.lang.ops

/**
 * Computation mode for the Mikrograd annotation.
 * This determines whether to use ForwardValue (INFERENCE) or BackwardValue (TRAINING).
 */
public enum class ComputationMode {
    /**
     * Inference mode uses ForwardValue which doesn't track gradients.
     * This is more memory-efficient when only forward pass is needed.
     */
    INFERENCE,

    /**
     * Training mode uses BackwardValue which tracks gradients for backpropagation.
     * This is necessary when gradient computation is needed.
     */
    TRAINING
}

/**
 * Annotation for functions that should be processed by the Mikrograd KSP processor.
 * The processor will generate optimized code for the function based on the computation mode.
 * 
 * @param mode The computation mode to use (INFERENCE or TRAINING)
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.SOURCE)
public annotation class TensorOp(val mode: ComputationMode = ComputationMode.INFERENCE)

/**
 * Annotation to mark classes or functions as not implemented for specific backends.
 * 
 * @param backends List of backend names where this feature is not implemented
 */
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.SOURCE)
public annotation class NotImplemented(vararg val backends: String)

/**
 * Annotation to mark classes or functions as in progress for specific backends.
 * 
 * @param backends List of backend names where this feature is in progress
 * @param owner The person or team responsible for the implementation
 * @param issue URL or identifier for the tracking issue
 */
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.SOURCE)
public annotation class InProgress(
    vararg val backends: String,
    val owner: String = "",
    val issue: String = ""
)

/**
 * Marks a class as a concrete compute backend implementation of the
 * `TensorOps` interface. The docs KSP processor uses this to derive the
 * `statusByBackend` map for each operator automatically, so adding a new
 * backend is one annotation instead of N hand-edits to `@InProgress`.
 *
 * @param id Stable identifier used as a column key in the ops status
 *   matrix (e.g. `"cpu"`, `"apple"`, `"wasm"`, `"cuda"`). Keep it short
 *   and lowercase.
 * @param displayName Human-readable label for rendered tables. Defaults
 *   to [id] if left empty.
 * @param internal Marks the backend as internal-only — a shape/dtype
 *   sentinel, test double, or profiling stub that should never appear in
 *   user-facing docs or coverage matrices. `VoidTensorOps` is the canonical
 *   example: it exists so the KMP build and shape propagation work without
 *   a real compute backend, but it has no runtime on any target.
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
public annotation class Backend(
    val id: String,
    val displayName: String = "",
    val internal: Boolean = false,
)
