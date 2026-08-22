package sk.ainet.lang.memory

/**
 * Marks the SKEEP-003 memory-architecture API (`sk.ainet.lang.memory`) that is still being
 * shaped through milestones M0–M1 (`Format`, `AllocationSpec`, later `Storage`, `Scope`,
 * `TensorView`, …). The types are usable and tested, but their shape may still change before the
 * compatibility promise applies; opt in explicitly with `@OptIn(ExperimentalMemoryApi::class)`.
 */
@RequiresOptIn(
    message = "SKEEP-003 memory-architecture API: usable, but may change until milestone M1 is complete.",
    level = RequiresOptIn.Level.WARNING,
)
@Retention(AnnotationRetention.BINARY)
@Target(
    AnnotationTarget.CLASS, AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY,
    AnnotationTarget.TYPEALIAS, AnnotationTarget.CONSTRUCTOR,
)
public annotation class ExperimentalMemoryApi
