package sk.ainet.lang.trace

/**
 * Annotation to mark TensorOps methods that require an adjoint (gradient) implementation.
 * 
 * When applied to a method in an interface annotated with @GenerateTracingWrapper,
 * the KSP processor will ensure that this operation is tracked as differentiable.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.SOURCE)
public annotation class Diff(
    val ruleName: String = ""
)
