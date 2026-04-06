package sk.ainet.lang.tensor.storage

/**
 * Declares placement intent for a tensor parameter or property.
 *
 * The [MemoryPlanner] reads these annotations (via reflection or codegen)
 * to decide where tensors should be allocated. This expresses *intent*,
 * not a hard guarantee — the planner may fall back if the target is
 * unavailable and [requirement] is [Requirement.PREFERRED].
 *
 * Example:
 * ```kotlin
 * @Place(device = DeviceKind.GPU, memory = MemoryDomain.DEVICE_LOCAL)
 * val projectionWeight: Tensor<FP32, Float>
 * ```
 */
@Target(AnnotationTarget.PROPERTY, AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class Place(
    val device: DeviceKind = DeviceKind.AUTO,
    val memory: MemoryDomain = MemoryDomain.HOST_HEAP,
    val requirement: Requirement = Requirement.PREFERRED
)

/**
 * Marks a tensor as an immutable weight that should be file-backed
 * (memory-mapped) when possible.
 *
 * Equivalent to `@Place(device = CPU, memory = MMAP_FILE)` with
 * [Residency.PERSISTENT]. The planner treats these tensors as
 * read-only and long-lived, preferring OS-paged file access over
 * heap allocation.
 *
 * Example:
 * ```kotlin
 * @Weights
 * val embeddings: Tensor<FP32, Float>
 *
 * @Weights(memory = MemoryDomain.HOST_HEAP)  // force heap for small weights
 * val biasVector: Tensor<FP32, Float>
 * ```
 */
@Target(AnnotationTarget.PROPERTY, AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class Weights(
    val memory: MemoryDomain = MemoryDomain.MMAP_FILE
)
