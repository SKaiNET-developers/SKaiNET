package sk.ainet.buildlogic.kmp

import com.android.build.api.dsl.KotlinMultiplatformAndroidLibraryTarget
import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.artifacts.VersionCatalog
import org.gradle.api.artifacts.VersionCatalogsExtension
import org.gradle.api.plugins.ExtensionAware
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.targets.js.dsl.KotlinJsSubTargetDsl
import sk.ainet.buildlogic.npm.NpmPinsMarker

private const val ANDROID_KMP_PLUGIN_ID = "com.android.kotlin.multiplatform.library"
private const val NPM_PINS_PLUGIN_ID = "sk.ainet.npm-pins"
private const val SHARED_KARMA_CONFIG_DIR = "gradle/karma.config.d"

/**
 * SKaiNET's standard Kotlin Multiplatform module setup.
 *
 * Replaces the target list, `android { }` block, `explicitApi()`, `kotlin-test` wiring
 * and Karma hardening that every library module used to copy by hand.
 *
 * ```kotlin
 * plugins {
 *     id("sk.ainet.multiplatform")
 *     alias(libs.plugins.androidMultiplatformLibrary)   // opt into Android
 * }
 *
 * skainet {
 *     namespace = "sk.ainet.pipeline"
 * }
 * ```
 *
 * Configuration is split in two, for reasons explained on [SkainetTargets]: which
 * platforms are built comes from the `skainet.targets` Gradle property (readable while
 * the plugin is applied, which is when targets must be created); everything else lives
 * in the [SkainetMultiplatformExtension] DSL and is applied after evaluation.
 */
class SkainetMultiplatformPlugin : Plugin<Project> {

    override fun apply(project: Project) {
        val extension = project.extensions.create("skainet", SkainetMultiplatformExtension::class.java)
        val targets = SkainetTargets.from(project)

        // Checked before anything is configured, so a misconfigured build fails on the
        // reason rather than on a symptom further down.
        if (targets.web) requireNpmPins(project)

        project.pluginManager.apply("org.jetbrains.kotlin.multiplatform")
        val kotlin = project.extensions.getByType(KotlinMultiplatformExtension::class.java)

        // Eager: targets must exist before the module's own `kotlin { }` block runs.
        configureTargets(project, kotlin, targets)

        // Deferred: needs the values the build script sets in `skainet { }`.
        project.afterEvaluate { applyExtension(this, kotlin, extension) }
    }

    private fun applyExtension(
        project: Project,
        kotlin: KotlinMultiplatformExtension,
        extension: SkainetMultiplatformExtension,
    ) {
        if (extension.explicitApi) kotlin.explicitApi()

        if (extension.expectActualClasses) {
            kotlin.targets.configureEach {
                compilations.configureEach {
                    compileTaskProvider.configure {
                        compilerOptions.freeCompilerArgs.add("-Xexpect-actual-classes")
                    }
                }
            }
        }

        if (extension.kotlinTestInCommonTest) {
            val kotlinTest = project.versionCatalog().findLibrary("kotlin-test").orElseThrow {
                GradleException("[skainet-multiplatform] Version catalog is missing the 'kotlin-test' library alias")
            }
            kotlin.sourceSets.named("commonTest").configure {
                dependencies { implementation(kotlinTest) }
            }
        }

        configureAndroid(project, kotlin, extension)
    }

    /**
     * Fails a `js`/`wasmJs` module when the root project does not apply
     * `sk.ainet.npm-pins`.
     *
     * Without the root plugin the npm pins are simply absent: no Yarn `resolutions` are
     * written, the lockfiles drift back to whatever the transitive ranges resolve to, and
     * `verifyNpmPins` does not exist to catch it. That is a silent security regression,
     * so it fails loudly at configuration time instead.
     *
     * The check reads a build-scoped service rather than `rootProject.pluginManager`, so
     * it stays legal under isolated projects — see [NpmPinsMarker].
     */
    private fun requireNpmPins(project: Project) {
        if (NpmPinsMarker.isRegistered(project)) return

        throw GradleException(
            "[skainet-multiplatform] ${project.path} builds js/wasmJs targets, but " +
                    "$NPM_PINS_PLUGIN_ID is not applied to the root project, so npm pins " +
                    "are not in force. Add it to the root build.gradle.kts:\n" +
                    "    plugins { alias(libs.plugins.skainet.npmPins) }"
        )
    }

    /**
     * Fills in the `android { }` body. The AGP KMP plugin creates the target itself when
     * it is applied in the module's `plugins { }` block, so this only sets values — which
     * is why it can run after evaluation, ahead of AGP's own `afterEvaluate`.
     */
    private fun configureAndroid(
        project: Project,
        kotlin: KotlinMultiplatformExtension,
        extension: SkainetMultiplatformExtension,
    ) {
        if (!project.pluginManager.hasPlugin(ANDROID_KMP_PLUGIN_ID)) return

        val namespace = extension.namespace
            ?: throw GradleException(
                "[skainet-multiplatform] ${project.path} applies $ANDROID_KMP_PLUGIN_ID, " +
                        "so it must declare a namespace: skainet { namespace = \"sk.ainet.…\" }"
            )

        val catalog = project.versionCatalog()
        val android = (kotlin as ExtensionAware).extensions
            .getByName("android") as KotlinMultiplatformAndroidLibraryTarget

        android.namespace = namespace
        android.compileSdk = catalog.requiredVersion("android-compileSdk").toInt()
        android.minSdk = catalog.requiredVersion("android-minSdk").toInt()
        android.compilerOptions { jvmTarget.set(extension.androidJvmTarget) }
    }

    @OptIn(ExperimentalWasmDsl::class)
    private fun configureTargets(
        project: Project,
        kotlin: KotlinMultiplatformExtension,
        targets: SkainetTargets,
    ) {
        if (targets.jvm) kotlin.jvm()

        if (targets.js) {
            kotlin.js {
                browser { hardenBrowserTests(project) }
            }
        }

        if (targets.wasmJs) {
            kotlin.wasmJs {
                browser { hardenBrowserTests(project) }
                if (targets.wasmJsExecutable) binaries.executable()
            }
        }

        if (targets.wasmWasi) kotlin.wasmWasi { nodejs() }

        if (targets.apple) {
            kotlin.iosArm64()
            kotlin.iosSimulatorArm64()
            kotlin.macosArm64()
        }

        if (targets.linux) {
            kotlin.linuxX64()
            kotlin.linuxArm64()
        }

        if (targets.androidNative) {
            kotlin.androidNativeArm32()
            kotlin.androidNativeArm64()
        }

        if (targets.mingw) {
            kotlin.mingwX64()
        }
    }

    /**
     * Points Karma at the repository-wide config directory instead of a per-module
     * `karma.config.d`. The shared config raises capture/disconnect timeouts so that
     * parallel `allTests` runs do not fail on a starved headless browser.
     *
     * `useChromeHeadless()` is not redundant: KGP installs its own default only when no
     * test framework has been set yet (`KotlinBrowserJsIr.configureDefaultTestFramework`),
     * and `useKarma` replaces the framework wholesale. Dropping it leaves the task with
     * "No browsers configured for jsBrowserTest".
     */
    private fun KotlinJsSubTargetDsl.hardenBrowserTests(project: Project) {
        val sharedConfigDir = project.rootProject.file(SHARED_KARMA_CONFIG_DIR)
        testTask {
            useKarma {
                useChromeHeadless()
                useConfigDirectory(sharedConfigDir)
            }
        }
    }

    private fun Project.versionCatalog(): VersionCatalog =
        extensions.getByType(VersionCatalogsExtension::class.java).named("libs")

    private fun VersionCatalog.requiredVersion(alias: String): String =
        findVersion(alias)
            .orElseThrow { GradleException("[skainet-multiplatform] Version catalog is missing the '$alias' version alias") }
            .requiredVersion
}
