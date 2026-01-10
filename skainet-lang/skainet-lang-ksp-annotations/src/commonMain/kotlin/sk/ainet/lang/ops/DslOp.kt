package sk.ainet.lang.ops

/**
 * Annotation for functions that are part of the SKaiNET DSL.
 * These are typically composite operations built from primitive tensor operations.
 *
 * @param category The category of the operation (e.g., "Similarity", "Activation", "Loss").
 * @param description A brief description of what the operation does.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.SOURCE)
public annotation class DslOp(
    val category: String = "",
    val description: String = ""
)
