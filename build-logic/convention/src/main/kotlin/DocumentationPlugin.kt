import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.Action

class DocumentationPlugin : Plugin<Project> {
    override fun apply(project: Project) {
        val extension = project.extensions.create("documentation", DocumentationExtension::class.java, project)
        
        project.tasks.register("generateDocs", GenerateDocumentationTask::class.java, object : Action<GenerateDocumentationTask> {
            override fun execute(task: GenerateDocumentationTask) {
                task.group = "documentation"
                task.description = "Generate documentation from KSP metadata"
                
                task.inputFile.set(extension.inputFile)
                task.outputDirectory.set(extension.outputDirectory)
                task.templateDirectory.set(extension.templateDirectory)
                task.format.set(extension.format)
                task.includeBackendStatus.set(extension.includeBackendStatus)
                task.generateIndex.set(extension.generateIndex)
                task.backendStatusFiles.from(extension.backendStatusFiles)
                task.hloCoverageFile.set(extension.hloCoverageFile)
                task.backwardCoverageFile.set(extension.backwardCoverageFile)
                task.issueBaseUrl.set(extension.issueBaseUrl)
            }
        })

        // Kernel × platform matrix — kernel-side analogue of generateDocs. Renders the
        // registry-introspected kernel-support.json into an Antora .adoc.
        project.tasks.register("generateKernelMatrix", GenerateKernelMatrixTask::class.java, object : Action<GenerateKernelMatrixTask> {
            override fun execute(task: GenerateKernelMatrixTask) {
                task.group = "documentation"
                task.description = "Render the kernel × platform support matrix from kernel-support.json"
                task.inputFile.set(extension.kernelInputFile)
                task.outputFile.set(extension.kernelOutputFile)
            }
        })

        // Guard: every NotImplementedError / TODO( in production code is either inside a
        // @NotImplemented / @InProgress declaration (and so in the coverage matrix) or allowlisted.
        project.tasks.register("checkUnmarkedStubs", CheckUnmarkedStubsTask::class.java, object : Action<CheckUnmarkedStubsTask> {
            override fun execute(task: CheckUnmarkedStubsTask) {
                task.group = "verification"
                task.description = "Fail on NotImplementedError / TODO( stubs that are neither @NotImplemented nor allowlisted"
                val root = project.rootProject.layout.projectDirectory
                // Per-project src/ trees (not the repo root) so the scan never overlaps other
                // tasks' outputs such as the generated docs pages.
                project.rootProject.allprojects.forEach { p ->
                    task.sources.from(p.fileTree("src") { include("**/*.kt") })
                }
                task.sources.from(root.dir("build-logic").asFileTree.matching {
                    include("**/src/**/*.kt")
                    exclude("**/build/**", "**/.gradle/**")
                })
                task.rootDirectory.set(root)
                task.allowlistFile.set(root.file("config/stub-allowlist.txt"))
                task.reportFile.set(project.layout.buildDirectory.file("reports/stubs/unmarked-stubs.txt"))
            }
        })

        // Guard (SKEEP-003): ops choose kernels from a view's Format, not by downcasting to a
        // concrete TensorData class. Today's sites are allowlisted; the list may only shrink.
        project.tasks.register("checkTensorDataDowncasts", CheckTensorDataDowncastsTask::class.java, object : Action<CheckTensorDataDowncastsTask> {
            override fun execute(task: CheckTensorDataDowncastsTask) {
                task.group = "verification"
                task.description = "Fail on downcasts to concrete TensorData classes that are not in the allowlist"
                val root = project.rootProject.layout.projectDirectory
                project.rootProject.allprojects.forEach { p ->
                    task.sources.from(p.fileTree("src") { include("**/*.kt") })
                }
                task.rootDirectory.set(root)
                task.allowlistFile.set(root.file("config/tensor-data-downcast-allowlist.txt"))
                task.reportFile.set(project.layout.buildDirectory.file("reports/downcasts/tensor-data-downcasts.txt"))
            }
        })

        // Register schema validation task in plugin (migrated from skainet-lang-export-ops)
        val validateTaskProvider = project.tasks.register("validateOperatorSchema", SchemaValidationTask::class.java, object : Action<SchemaValidationTask> {
            override fun execute(task: SchemaValidationTask) {
                task.group = "verification"
                task.description = "Validate generated operators.json files against the JSON schema"
                // By default search from the root project dir to find all operators.json
                task.searchDirectory.set(project.rootProject.layout.projectDirectory)
            }
        })
    }
}