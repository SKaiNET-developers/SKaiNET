package sk.ainet.buildlogic.root

import org.gradle.api.Action
import org.gradle.api.model.ObjectFactory
import sk.ainet.buildlogic.npm.NpmPinsExtension
import javax.inject.Inject

/**
 * The `skainet { }` block contributed by SKaiNET's **root-project** conventions.
 *
 * Deliberately shares its name with the module-level block created by
 * `sk.ainet.multiplatform`, so a build script author sees one SKaiNET namespace
 * regardless of where they are. The types differ because the concerns do: a module
 * configures its own compilation, the root configures things that are global to the
 * build and have nowhere else to live.
 *
 * ```kotlin
 * // root build.gradle.kts
 * skainet {
 *     npmPins {
 *         pin("ws", libs.versions.npm.ws)
 *     }
 * }
 * ```
 *
 * Nest further root-level conventions here rather than adding top-level extensions;
 * [sk.ainet.buildlogic.root.SkainetRootExtension.Companion.findOrCreate] lets several
 * plugins contribute to the same block.
 */
abstract class SkainetRootExtension @Inject constructor(objects: ObjectFactory) {

    /**
     * npm packages forced onto an exact version across every Kotlin/JS and Kotlin/Wasm
     * lockfile. Populated by `sk.ainet.npm-pins`; see [NpmPinsExtension].
     */
    val npmPins: NpmPinsExtension = objects.newInstance(NpmPinsExtension::class.java)

    /** Configures [npmPins]. */
    fun npmPins(action: Action<in NpmPinsExtension>) {
        action.execute(npmPins)
    }

    companion object {

        const val NAME: String = "skainet"

        /**
         * Returns the root `skainet { }` extension, creating it if this is the first
         * root convention plugin to ask.
         *
         * @throws IllegalStateException if something else already claimed the name — most
         * likely `sk.ainet.multiplatform` applied to the root project, which is not
         * supported: the root project is not a library module.
         */
        fun findOrCreate(project: org.gradle.api.Project): SkainetRootExtension {
            val existing = project.extensions.findByName(NAME) ?: return project.extensions
                .create(NAME, SkainetRootExtension::class.java)

            return existing as? SkainetRootExtension ?: error(
                "[skainet] ${project.path} already has a '$NAME' extension of type " +
                        "${existing.javaClass.name}. The root-project '$NAME { }' block cannot " +
                        "coexist with it — is sk.ainet.multiplatform applied to the root project?"
            )
        }
    }
}
