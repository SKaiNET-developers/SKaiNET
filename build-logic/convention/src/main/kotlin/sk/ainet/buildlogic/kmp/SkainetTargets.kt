package sk.ainet.buildlogic.kmp

import org.gradle.api.InvalidUserDataException
import org.gradle.api.Project

/**
 * Which platforms a module targets.
 *
 * ## Why this is a Gradle property and not part of `skainet { }`
 *
 * Kotlin targets have to exist before the module's own `kotlin { }` block runs: the
 * source-set convention accessors (`jvmMain`, `iosArm64Main`, …) create source sets on
 * access, and KGP then refuses to attach a compilation to a source set that already
 * exists —
 *
 * ```
 * ⛔ The compilation 'main' cannot be created after the source set 'jvmMain'
 * ```
 *
 * A `skainet { }` block is evaluated as part of the build script, so anything declared
 * there is only known once it is already too late to create targets. Gradle properties,
 * on the other hand, are readable while the plugin is being applied. Everything that is
 * *not* structural stays in [SkainetMultiplatformExtension].
 *
 * ## Usage
 *
 * Most modules want the default set and declare nothing. A module that differs adds one
 * line to its own `gradle.properties`:
 *
 * ```properties
 * skainet.targets=jvm
 * skainet.wasmJs.executable=true
 * ```
 *
 * `skainet.targets` is a comma-separated list of the names below. `none` selects an
 * empty set, for modules that declare every target themselves.
 */
internal data class SkainetTargets(
    val jvm: Boolean,
    val js: Boolean,
    val wasmJs: Boolean,
    val wasmWasi: Boolean,
    /** `iosArm64`, `iosSimulatorArm64`, `macosArm64`. */
    val apple: Boolean,
    /** `linuxX64`, `linuxArm64`. */
    val linux: Boolean,
    /** `androidNativeArm32`, `androidNativeArm64` — vendor backends linking device libraries. */
    val androidNative: Boolean,
    val wasmJsExecutable: Boolean,
) {
    val web: Boolean get() = js || wasmJs

    companion object {
        const val TARGETS_PROPERTY = "skainet.targets"
        const val WASM_JS_EXECUTABLE_PROPERTY = "skainet.wasmJs.executable"

        private val DEFAULT = setOf("jvm", "js", "wasmJs", "wasmWasi", "apple", "linux")
        private val KNOWN = DEFAULT + setOf("androidNative")

        fun from(project: Project): SkainetTargets {
            // findProperty, not providers.gradleProperty: as of Gradle 9 the provider API
            // only exposes build-level properties, and these are declared per module.
            val declared = project.findProperty(TARGETS_PROPERTY)?.toString()
            val selected = when {
                declared == null -> DEFAULT
                declared.isBlank() || declared.trim() == "none" -> emptySet()
                else -> declared.split(",").map { it.trim() }.filter { it.isNotEmpty() }.toSet()
            }

            val unknown = selected - KNOWN
            if (unknown.isNotEmpty()) {
                throw InvalidUserDataException(
                    "[skainet-multiplatform] ${project.path}: unknown $TARGETS_PROPERTY entries " +
                            "${unknown.sorted()}. Known values: ${KNOWN.sorted()}, or 'none'."
                )
            }

            return SkainetTargets(
                jvm = "jvm" in selected,
                js = "js" in selected,
                wasmJs = "wasmJs" in selected,
                wasmWasi = "wasmWasi" in selected,
                apple = "apple" in selected,
                linux = "linux" in selected,
                androidNative = "androidNative" in selected,
                wasmJsExecutable = project.findProperty(WASM_JS_EXECUTABLE_PROPERTY)?.toString().toBoolean(),
            )
        }
    }
}
