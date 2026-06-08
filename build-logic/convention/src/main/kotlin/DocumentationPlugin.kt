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