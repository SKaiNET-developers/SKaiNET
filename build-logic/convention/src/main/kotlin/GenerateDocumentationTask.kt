import models.*
import kotlinx.serialization.json.Json
import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.*
import java.io.File
import java.time.Instant
import java.time.format.DateTimeFormatter

@CacheableTask
abstract class GenerateDocumentationTask : DefaultTask() {
    
    @get:InputFile
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val inputFile: RegularFileProperty
    
    @get:OutputDirectory
    abstract val outputDirectory: DirectoryProperty
    
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    @get:Optional
    abstract val templateDirectory: DirectoryProperty
    
    @get:Input
    abstract val format: Property<DocumentationFormat>
    
    @get:Input
    @get:Optional
    abstract val includeBackendStatus: Property<Boolean>
    
    @get:Input
    @get:Optional
    abstract val generateIndex: Property<Boolean>
    
    @TaskAction
    fun generateDocumentation() {
        val input = inputFile.get().asFile
        val output = outputDirectory.get().asFile
        
        logger.lifecycle("📚 Generating documentation from: ${input.absolutePath}")
        logger.lifecycle("📂 Output directory: ${output.absolutePath}")
        
        val jsonContent = input.readText()
        val json = Json { ignoreUnknownKeys = true }
        val module = json.decodeFromString<OperatorDocModule>(jsonContent)
        
        when (format.get()) {
            DocumentationFormat.ASCIIDOC -> generateAsciidoc(module, output)
            DocumentationFormat.MARKDOWN -> generateMarkdown(module, output)
            DocumentationFormat.HTML -> generateHtml(module, output)
        }
        
        logger.lifecycle("✅ Documentation generation completed!")
        logger.lifecycle("📖 Generated docs can be found at: ${output.absolutePath}")
        if (generateIndex.getOrElse(true)) {
            val indexFile = File(output, "index.adoc")
            if (indexFile.exists()) {
                logger.lifecycle("🏠 Main index file: ${indexFile.absolutePath}")
            }
        }
    }
    
    private fun generateAsciidoc(module: OperatorDocModule, outputDir: File) {
        outputDir.mkdirs()
        
        if (generateIndex.getOrElse(true)) {
            generateMainIndex(module, outputDir)
        }
        
        module.operators.forEach { operator ->
            generateOperatorPage(operator, module, outputDir)
        }
    }
    
    private fun generateMarkdown(module: OperatorDocModule, outputDir: File) {
        // TODO: Implement markdown generation
        throw NotImplementedError("Markdown generation not implemented yet")
    }
    
    private fun generateHtml(module: OperatorDocModule, outputDir: File) {
        // TODO: Implement HTML generation  
        throw NotImplementedError("HTML generation not implemented yet")
    }
    
    private fun generateMainIndex(module: OperatorDocModule, outputDir: File) {
        val indexFile = File(outputDir, "index.adoc")
        // When the output directory sits under an Antora module's
        // `modules/<name>/pages/` tree, xrefs in the emitted index
        // must be resolved relative to that `pages/` root, not the
        // current file. Auto-derive the prefix from the output path
        // so the generator works both with Antora and with flat doc
        // layouts (empty prefix -> bare filenames, the original
        // behavior).
        val xrefPrefix = deriveAntoraXrefPrefix(outputDir)
        indexFile.writeText(buildString {
            appendLine("= AI-NET Operators Reference")
            appendLine("")
            appendLine("Generated from version `${module.version}` on ${formatTimestamp(module.timestamp)}")
            appendLine("")
            appendLine("== Operators by Modality")
            appendLine("")

            val operatorsByModality = module.operators.groupBy { it.modality }
            operatorsByModality.forEach { (modality, operators) ->
                appendLine("=== ${modality.capitalize()}")
                appendLine("")
                operators.forEach { operator ->
                    appendLine("* xref:$xrefPrefix${operator.name.lowercase()}.adoc[${operator.name}]")
                }
                appendLine("")
            }
        })
    }

    /**
     * If [outputDir] lives under an Antora `modules/<name>/pages/...`
     * tree, return the path segment from `pages/` down to the output
     * directory, suffixed with `/`. Otherwise return an empty string,
     * so the generator emits bare-filename xrefs (the pre-Antora
     * behavior).
     *
     * Example:
     * ```
     * /repo/docs/modules/ROOT/pages/reference/operators/generated
     *                                → "reference/operators/generated/"
     * /repo/docs/operators/generated → ""
     * ```
     */
    private fun deriveAntoraXrefPrefix(outputDir: File): String {
        val path = outputDir.absolutePath.replace(File.separatorChar, '/')
        val marker = "/pages/"
        val idx = path.indexOf(marker)
        if (idx < 0) return ""
        val tail = path.substring(idx + marker.length)
        return if (tail.isEmpty()) "" else "$tail/"
    }
    
    private fun generateOperatorPage(operator: OperatorDoc, module: OperatorDocModule, outputDir: File) {
        val operatorFile = File(outputDir, "${operator.name.lowercase()}.adoc")
        operatorFile.writeText(buildString {
            appendLine("= ${operator.name}")
            appendLine("")
            appendLine("Package: `${operator.packageName}`")
            appendLine("")
            appendLine("Modality: ${operator.modality.capitalize()}")
            appendLine("")
            
            operator.functions.forEach { function ->
                generateFunctionSection(function, this)
            }
        })
    }
    
    private fun generateFunctionSection(function: FunctionDoc, builder: StringBuilder) {
        builder.apply {
            appendLine("== ${function.name}")
            appendLine("")
            appendLine("=== Signature")
            appendLine("")
            appendLine("[source,kotlin]")
            appendLine("----")
            appendLine(function.signature)
            appendLine("----")
            appendLine("")
            
            if (function.parameters.isNotEmpty()) {
                appendLine("=== Parameters")
                appendLine("")
                function.parameters.forEach { param ->
                    appendLine("* `${param.name}: ${param.type}`")
                    if (param.description.isNotEmpty()) {
                        appendLine("  ${param.description}")
                    }
                }
                appendLine("")
            }
            
            appendLine("=== Return Type")
            appendLine("")
            appendLine("`${function.returnType}`")
            appendLine("")
            
            if (includeBackendStatus.getOrElse(true) && function.statusByBackend.isNotEmpty()) {
                generateBackendStatusTable(function, this)
            }
            
            if (function.notes.isNotEmpty()) {
                appendLine("=== Notes")
                appendLine("")
                function.notes.forEach { note ->
                    appendLine("TIP: *${note.backend}*: ${note.message}")
                    appendLine("")
                }
            }
            
            appendLine("")
        }
    }
    
    private fun generateBackendStatusTable(function: FunctionDoc, builder: StringBuilder) {
        builder.apply {
            appendLine("=== Backend Support")
            appendLine("")
            appendLine("[cols=\"1,1,3\", options=\"header\"]")
            appendLine("|===")
            appendLine("| Backend | Status | Notes")
            
            function.statusByBackend.forEach { (backend, status) ->
                val formattedStatus = formatStatus(status)
                val notes = function.notes
                    .filter { it.backend.equals(backend, ignoreCase = true) }
                    .joinToString("; ") { it.message }
                
                appendLine("| $backend | $formattedStatus | ${notes.ifEmpty { "-" }}")
            }
            
            appendLine("|===")
            appendLine("")
        }
    }
    
    private fun formatStatus(status: String): String {
        return when (status.lowercase()) {
            "supported" -> "✅ Supported"
            "partial" -> "⚠️ Partial"
            "not_supported" -> "❌ Not Supported"
            "planned" -> "📋 Planned"
            else -> status
        }
    }
    
    private fun formatTimestamp(timestamp: String): String {
        return try {
            // Simple timestamp formatting - just return the first 10 characters (date part)
            if (timestamp.length >= 10) timestamp.substring(0, 10) else timestamp
        } catch (e: Exception) {
            timestamp
        }
    }
    
    private fun String.capitalize(): String = 
        this.lowercase().replaceFirstChar { it.uppercase() }
}