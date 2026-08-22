package sk.ainet.buildlogic.maven

import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.artifacts.Configuration
import org.gradle.api.artifacts.component.ModuleComponentIdentifier
import org.gradle.api.artifacts.result.ResolvedComponentResult
import org.gradle.api.artifacts.result.ResolvedDependencyResult
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.MapProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.Internal
import org.gradle.api.tasks.TaskAction
import org.gradle.work.DisableCachingByDefault

/**
 * Fails when a pin declared in `skainet { mavenPins { } }` no longer matches what
 * actually resolves in the dependency graph.
 *
 * Unlike Yarn, the Maven/JVM world has no committed lockfile to diff — `verifyMavenPins`
 * walks the live resolved dependency graph of every `*CompileClasspath`/`*RuntimeClasspath`
 * configuration across every subproject instead (metadata only, no artifact download).
 * That live-resolution walk can't be expressed as configuration-cache-safe task inputs
 * (a [Configuration] isn't serializable, and wrapping per-configuration resolution
 * results in ordinary `Provider`s still forces resolution outside the execution lock
 * configuration cache's store phase runs under), so this task opts out of it entirely.
 */
@DisableCachingByDefault(because = "Inspects live dependency-resolution results; caching costs more than it saves")
abstract class VerifyMavenPinsTask : DefaultTask() {

    init {
        notCompatibleWithConfigurationCache(
            "verifyMavenPins resolves dependency configurations directly against the live graph"
        )
    }

    /** `"group:artifact"` -> pinned version. */
    @get:Input
    abstract val pins: MapProperty<String, String>

    @get:Input
    abstract val failOnMissingModule: Property<Boolean>

    /** Every `*CompileClasspath`/`*RuntimeClasspath` configuration across every subproject. */
    @get:Internal
    abstract val configurations: ListProperty<Configuration>

    @TaskAction
    fun verify() {
        val declaredPins = pins.get()
        if (declaredPins.isEmpty()) {
            logger.lifecycle("[maven-pins] No maven pins declared; nothing to verify.")
            return
        }

        val resolvedVersions = mutableMapOf<String, String>()
        val visited = mutableSetOf<String>()
        var checkedConfigurations = 0

        fun walk(component: ResolvedComponentResult) {
            if (!visited.add(component.id.displayName)) return
            val id = component.id
            if (id is ModuleComponentIdentifier) {
                val coordinate = "${id.group}:${id.module}"
                if (coordinate in declaredPins) {
                    resolvedVersions[coordinate] = id.version
                }
            }
            component.dependencies.forEach { dependency ->
                if (dependency is ResolvedDependencyResult) {
                    walk(dependency.selected)
                }
            }
        }

        configurations.get().forEach { configuration ->
            checkedConfigurations++
            walk(configuration.incoming.resolutionResult.root)
        }

        val mismatches = declaredPins.mapNotNull { (coordinate, pinned) ->
            val resolved = resolvedVersions[coordinate]
            if (resolved != null && resolved != pinned) {
                "$coordinate resolved to $resolved, pinned to $pinned"
            } else {
                null
            }
        }

        val missing = (declaredPins.keys - resolvedVersions.keys).sorted()
        if (missing.isNotEmpty()) {
            val message = buildString {
                appendLine("[maven-pins] Pinned but never resolved by any subproject's dependency graph:")
                missing.forEach { appendLine("  - $it") }
                append(
                    "The pin may be stale — drop it from gradle/libs.versions.toml if the " +
                            "dependency that pulled it in transitively is gone for good."
                )
            }
            if (failOnMissingModule.get()) throw GradleException(message) else logger.warn(message)
        }

        if (mismatches.isNotEmpty()) {
            throw GradleException(
                buildString {
                    appendLine("[maven-pins] Resolved dependency graph does not honour the declared maven pins:")
                    mismatches.sorted().forEach { appendLine("  - $it") }
                }
            )
        }

        logger.lifecycle(
            "[maven-pins] ${declaredPins.size} pin(s) verified across $checkedConfigurations configuration(s)."
        )
    }
}
