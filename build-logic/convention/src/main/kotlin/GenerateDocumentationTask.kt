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

        // Sibling cross-backend coverage matrix. Lives one level above
        // the per-operator pages so a single URL gives the whole
        // picture. Skipped when includeBackendStatus is disabled.
        if (includeBackendStatus.getOrElse(true)) {
            emitOpsStatusMatrix(module, outputDir)
        }
    }

    /**
     * Emit a single-page `ops-status-matrix.adoc` with rows of
     * operator.function pairs and columns of every backend that
     * appears in any function's `statusByBackend` map. Cells carry
     * the status emoji; a totals footer shows how many functions
     * each backend supports out of the total.
     *
     * Written to [outputDir].parentFile.parentFile so that, under the
     * Antora `reference/operators/generated/` layout, the matrix
     * lands at `reference/ops-status-matrix.adoc` — one navigable
     * click away from the operator index and with a stable URL.
     * Falls back to writing next to [outputDir] when the path
     * doesn't have the expected depth (flat layouts).
     */
    private fun emitOpsStatusMatrix(module: OperatorDocModule, outputDir: File) {
        val matrixDir = outputDir.parentFile?.parentFile ?: outputDir
        matrixDir.mkdirs()
        val matrixFile = File(matrixDir, "ops-status-matrix.adoc")

        // Collect every backend that appears anywhere, sorted so the
        // column order is stable across runs.
        val allBackends: List<String> = module.operators
            .flatMap { op -> op.functions.flatMap { it.statusByBackend.keys } }
            .toSortedSet()
            .toList()

        // Row view carries the per-function status plus DARC validation
        // signal so the matrix can render both in one pass.
        val partialsRoot = derivePartialsRoot(outputDir)
        data class Row(
            val operator: String,
            val function: FunctionDoc,
            val hasPartial: Boolean,
        )
        val rows: List<Row> = module.operators.flatMap { op ->
            op.functions.map { fn ->
                val relative = "ops/${op.name.lowercase()}/${fn.name.lowercase()}.adoc"
                val present = partialsRoot?.let { File(it, relative).isFile } == true
                Row(op.name, fn, present)
            }
        }

        matrixFile.writeText(buildString {
            appendLine("= Operator Coverage Matrix")
            appendLine(":description: Cross-backend status for every operator function in SKaiNET.")
            appendLine("")
            appendLine("Generated from `operators.json` version `${module.version}` on ${formatTimestamp(module.timestamp)}.")
            appendLine("")
            appendLine("Rows are `Operator.function` pairs. The `Validated` column shows whether the function's documentation has been DARC-validated by a reviewer (see xref:contributing/darc-workflow.adoc[DARC workflow]). Remaining columns are backends that appear in any function's `statusByBackend` map — a missing entry means the backend makes no claim about the function (treat it as \"unknown\", not \"not supported\").")
            appendLine("")
            if (rows.isEmpty() || allBackends.isEmpty()) {
                appendLine("NOTE: No backend status information found in the source data.")
                appendLine("")
                return@buildString
            }

            // Header: row label, validation column, then one column per backend.
            val colSpec = (listOf("2", "1") + List(allBackends.size) { "1" }).joinToString(",")
            appendLine("[cols=\"$colSpec\", options=\"header\"]")
            appendLine("|===")
            append("| Operator.function | Validated ")
            allBackends.forEach { append("| $it ") }
            appendLine("")
            appendLine("")

            rows.forEach { row ->
                append("| `${row.operator}.${row.function.name}` ")
                append("| ${darcCell(row.function, row.hasPartial)} ")
                allBackends.forEach { backend ->
                    val raw = row.function.statusByBackend[backend]
                    val cell = if (raw == null) "—" else shortStatus(raw)
                    append("| $cell ")
                }
                appendLine("")
            }

            // Totals footer: validated count followed by per-backend done count.
            appendLine("")
            val validatedCount = rows.count { it.function.validated }
            append("| *Done* | *$validatedCount / ${rows.size}* ")
            allBackends.forEach { backend ->
                val n = rows.count { isDone(it.function.statusByBackend[backend]) }
                append("| *$n / ${rows.size}* ")
            }
            appendLine("")
            appendLine("|===")
            appendLine("")
            appendLine("Per-function detail including notes lives in xref:reference/operators/generated/index.adoc[Operator reference].")
        })
    }

    /**
     * One-line DARC validation badge rendered above each function's
     * signature on the generated page.
     *
     * Three states, ordered from strongest to weakest signal:
     *   - validated: a reviewer (not the original author) has signed off
     *     on the partial prose. Carries the validator and date.
     *   - prose without validation: a partial exists but no
     *     `@DarcValidated` annotation backs it. Treated as a stub.
     *   - no prose: only auto-generated facts (signature, parameters,
     *     return type). The reader is told explicitly so they don't
     *     mistake terseness for completeness.
     *
     * The bracketed CSS class is a hook for the Antora UI bundle to
     * style the badge; the visible text is the contract.
     */
    private fun darcBadge(function: FunctionDoc, hasPartial: Boolean): String = when {
        function.validated -> {
            val on = function.validatedOn.ifBlank { "an unspecified date" }
            val by = function.validatedBy.ifBlank { "an unspecified reviewer" }
            "[.darc-validated]#✅ DARC-validated by $by on $on#"
        }
        hasPartial -> "[.darc-stub]#⚠ Prose present but not DARC-validated#"
        else -> "[.darc-none]#✖ Generated facts only (no human prose)#"
    }

    /**
     * Emoji-only DARC status for the coverage matrix column.
     * Matches the three branches of [darcBadge].
     */
    private fun darcCell(function: FunctionDoc, hasPartial: Boolean): String = when {
        function.validated -> "✅"
        hasPartial -> "⚠"
        else -> "✖"
    }

    /**
     * Short emoji-only rendering of a backend status, for use in the
     * compact matrix cells.
     *
     * The vocabulary covers both the planning-style strings
     * (`supported` / `partial` / `not_supported` / `planned`) and
     * the implementation-style strings the KSP processor actually
     * emits today (`implemented` / `in_progress` / `missing`).
     * Unknown values fall back to the raw string so the matrix
     * never silently hides a status the generator didn't anticipate.
     */
    private fun shortStatus(status: String): String = when (status.lowercase()) {
        "supported", "implemented", "done" -> "✅"
        "partial" -> "⚠️"
        "not_supported", "missing", "unsupported" -> "❌"
        "planned" -> "⏳"
        "in_progress", "wip" -> "🚧"
        else -> status
    }

    /**
     * Whether a status string counts toward the totals footer in
     * the ops-status matrix. Mirrors the "green check" branch of
     * [shortStatus] — any status rendered with ✅ is counted as
     * done.
     */
    private fun isDone(status: String?): Boolean = when (status?.lowercase()) {
        "supported", "implemented", "done" -> true
        else -> false
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
        val partialsRoot = derivePartialsRoot(outputDir)
        operatorFile.writeText(buildString {
            appendLine("= ${operator.name}")
            appendLine("")
            appendLine("Package: `${operator.packageName}`")
            appendLine("")
            appendLine("Modality: ${operator.modality.capitalize()}")
            appendLine("")

            operator.functions.forEach { function ->
                generateFunctionSection(operator, function, this, partialsRoot)
            }
        })
    }

    /**
     * If [outputDir] lives under an Antora `modules/<name>/pages/...` tree,
     * return the sibling `partials/` directory where hand-written prose
     * snippets live. Returns `null` for flat doc layouts, in which case the
     * generator assumes no partials exist and skips include directives.
     *
     * Antora resolves `partial$...` through its content catalog, and unlike
     * plain AsciiDoctor it does *not* honor the `optional` attribute for
     * missing resources — it logs an "unresolved include" error. So the
     * generator must only emit include directives for partials that are
     * actually present on disk.
     */
    private fun derivePartialsRoot(outputDir: File): File? {
        val path = outputDir.absolutePath.replace(File.separatorChar, '/')
        val marker = "/pages/"
        val idx = path.indexOf(marker)
        if (idx < 0) return null
        val moduleRoot = File(path.substring(0, idx))
        return File(moduleRoot, "partials")
    }

    /**
     * Per-function section layout fuses auto-derived facts (signature,
     * parameters, return type, backend matrix) with optional hand-written
     * prose pulled from a partial at
     * `partials/ops/<operator>/<function>.adoc`. The partial is sliced by
     * AsciiDoc tags — `math`, `intuition`, `examples`, `references` — so a
     * single file per function carries all the human content, and missing
     * tags render as empty via `optional`, keeping un-prosed ops valid.
     */
    private fun generateFunctionSection(
        operator: OperatorDoc,
        function: FunctionDoc,
        builder: StringBuilder,
        partialsRoot: File?,
    ) {
        val partialRelative = "ops/${operator.name.lowercase()}/${function.name.lowercase()}.adoc"
        val partialFile = partialsRoot?.let { File(it, partialRelative) }
        val hasPartial = partialFile?.isFile == true
        builder.apply {
            appendLine("== ${function.name}")
            appendLine("")
            appendLine(darcBadge(function, hasPartial))
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

            // Human prose: only emitted when a partial actually exists on
            // disk. Antora does not honor the `optional` attribute for
            // `partial$` resource refs, so emitting includes for missing
            // partials produces "unresolved include" errors on the
            // published site.
            if (hasPartial) {
                appendLine("=== Definition")
                appendLine("")
                appendLine("include::partial\$$partialRelative[tag=math,optional]")
                appendLine("")
                appendLine("=== Intuition")
                appendLine("")
                appendLine("include::partial\$$partialRelative[tag=intuition,optional]")
                appendLine("")
                appendLine("=== Examples")
                appendLine("")
                appendLine("include::partial\$$partialRelative[tag=examples,optional]")
                appendLine("")
            }

            if (function.notes.isNotEmpty()) {
                appendLine("=== Notes")
                appendLine("")
                function.notes.forEach { note ->
                    appendLine("TIP: *${note.backend}*: ${note.message}")
                    appendLine("")
                }
            }

            if (hasPartial) {
                appendLine("=== References")
                appendLine("")
                appendLine("include::partial\$$partialRelative[tag=references,optional]")
                appendLine("")
            }
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