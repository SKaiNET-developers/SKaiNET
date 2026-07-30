package sk.ainet.buildlogic.npm

import org.gradle.api.Project
import org.gradle.api.services.BuildService
import org.gradle.api.services.BuildServiceParameters

/**
 * Build-scoped marker announcing that `sk.ainet.npm-pins` is applied to the root project.
 *
 * ## Why a build service and not `rootProject.pluginManager.hasPlugin(...)`
 *
 * A module that builds `js`/`wasmJs` has to be able to tell whether pins are in force —
 * without them a security pin silently stops applying and the lockfiles drift on the
 * next refresh, with no failing task to notice it (`verifyNpmPins` would not exist
 * either). Reading the root project's plugin state from a subproject answers that, but
 * it is exactly the cross-project access Gradle's isolated-projects mode rejects.
 *
 * Build services are build-scoped by design and safe to look up from any project. The
 * registration is lazy — [NpmPinsMarker] is never instantiated, only registered — so
 * this costs nothing at execution time.
 *
 * The ordering the check relies on is guaranteed: the root build script is evaluated
 * before any subproject is configured, so a root-applied `sk.ainet.npm-pins` has always
 * registered by the time a module plugin asks.
 */
abstract class NpmPinsMarker : BuildService<BuildServiceParameters.None> {

    companion object {

        private const val NAME = "skainetNpmPinsMarker"

        /** Called by [NpmPinsPlugin] once it has verified it is on the root project. */
        fun register(project: Project) {
            project.gradle.sharedServices.registerIfAbsent(NAME, NpmPinsMarker::class.java) {}
        }

        /** Whether `sk.ainet.npm-pins` was applied to the root project of this build. */
        fun isRegistered(project: Project): Boolean =
            project.gradle.sharedServices.registrations.findByName(NAME) != null
    }
}
