package sk.ainet.lang.dag

/**
 * Annotation to mark interfaces for automatic Graph DSL extension generation via KSP.
 *
 * When applied to an interface (like TensorOps), the KSP processor will generate
 * extension functions for DagBuilder that allow using these operations in the DSL.
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
public annotation class GenerateGraphDsl
