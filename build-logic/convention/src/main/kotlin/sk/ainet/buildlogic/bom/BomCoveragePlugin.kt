package sk.ainet.buildlogic.bom

import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project

private const val PUBLISH_PLUGIN_ID = "com.vanniktech.maven.publish"

class BomCoveragePlugin : Plugin<Project> {
    override fun apply(project: Project) {
        require(project.plugins.hasPlugin("java-platform")) {
            "sk.ainet.transformers.bom-coverage requires the java-platform plugin to be applied first"
        }

        val ext = project.extensions.create(
            "bomCoverage",
            BomCoverageExtension::class.java,
        )
        ext.excludePublished.convention(emptySet())

        project.rootProject.subprojects
            .filter { it.path != project.path }
            .forEach { project.evaluationDependsOn(it.path) }

        project.afterEvaluate {
            val excluded = ext.excludePublished.get() + project.path
            val publishedPaths = project.rootProject.subprojects
                .filter { it.plugins.hasPlugin(PUBLISH_PLUGIN_ID) }
                .map { it.path }
                .filterNot { it in excluded }
                .sorted()

            if (publishedPaths.isEmpty()) {
                throw GradleException(
                    "[bom-coverage] No published subprojects found for ${project.path}. " +
                            "At least one sibling must apply '$PUBLISH_PLUGIN_ID'."
                )
            }

            project.dependencies.constraints {
                publishedPaths.forEach { add("api", project.project(it)) }
            }
        }
    }
}
