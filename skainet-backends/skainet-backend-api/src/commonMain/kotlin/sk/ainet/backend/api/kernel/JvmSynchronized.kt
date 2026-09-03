package sk.ainet.backend.api.kernel

/**
 * `kotlin.jvm.Synchronized` for common code: a no-op everywhere except the JVM and Android,
 * where it actualizes to the real annotation. The registries below are written once at
 * bootstrap and read from many threads by schedule workers (SKEEP-005); on targets without
 * shared-memory threads nothing needs guarding.
 */
@OptIn(ExperimentalMultiplatform::class)
@OptionalExpectation
@Target(AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY_GETTER, AnnotationTarget.PROPERTY_SETTER)
@Retention(AnnotationRetention.SOURCE)
internal expect annotation class JvmSynchronized()
