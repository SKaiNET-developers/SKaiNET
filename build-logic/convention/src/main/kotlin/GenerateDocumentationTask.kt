import models.*
import kotlinx.serialization.json.Json
import org.gradle.api.DefaultTask
import org.gradle.api.file.ConfigurableFileCollection
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

    /** Backend-module operators.json files whose `statusByBackend` maps are merged in. */
    @get:InputFiles
    @get:Optional
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val backendStatusFiles: ConfigurableFileCollection

    /** Op names the StableHLO converter lowers; drives the StableHLO column when present. */
    @get:InputFiles
    @get:Optional
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val hloCoverageFile: RegularFileProperty

    /** Op name → implemented / stubbed / missing; drives the Backward column when present. */
    @get:InputFiles
    @get:Optional
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val backwardCoverageFile: RegularFileProperty

    @get:Input
    @get:Optional
    abstract val issueBaseUrl: Property<String>
    
    @TaskAction
    fun generateDocumentation() {
        val input = inputFile.get().asFile
        val output = outputDirectory.get().asFile
        
        logger.lifecycle("📚 Generating documentation from: ${input.absolutePath}")
        logger.lifecycle("📂 Output directory: ${output.absolutePath}")
        
        val jsonContent = input.readText()
        val json = Json { ignoreUnknownKeys = true }
        val module = expandAllBackends(mergeBackendStatus(json.decodeFromString<OperatorDocModule>(jsonContent), json))
        
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
    
    /**
     * Merge `statusByBackend` and notes from backend-module operators.json files. Backend
     * modules see `TensorOps` only as a library class, so their SOURCE-retained annotations
     * (`@Diff`, `@DarcValidated`, `@Lowering`) are lost there — only status is taken from them.
     */
    private fun mergeBackendStatus(module: OperatorDocModule, json: Json): OperatorDocModule {
        val fragments = backendStatusFiles.files.filter { it.isFile }
        if (fragments.isEmpty()) return module
        val byFunction = mutableMapOf<Pair<String, String>, FunctionDoc>()
        fragments.forEach { file ->
            logger.lifecycle("🔗 Merging backend status from: ${file.absolutePath}")
            json.decodeFromString<OperatorDocModule>(file.readText()).operators.forEach { op ->
                op.functions.forEach { fn -> byFunction[op.name to fn.name] = fn }
            }
        }
        return module.copy(operators = module.operators.map { op ->
            op.copy(functions = op.functions.map { fn ->
                val extra = byFunction[op.name to fn.name] ?: return@map fn
                val status = fn.statusByBackend.toMutableMap()
                extra.statusByBackend.forEach { (backend, value) -> status[backend] = mergeStatus(status[backend], value) }
                fn.copy(statusByBackend = status, notes = fn.notes + extra.notes)
            })
        })
    }

    /** Expand a `"*"` status (marker without a backend list) onto every backend column. */
    private fun expandAllBackends(module: OperatorDocModule): OperatorDocModule {
        val backends = module.operators.flatMap { op -> op.functions.flatMap { it.statusByBackend.keys } }.toSet() - "*"
        return module.copy(operators = module.operators.map { op ->
            op.copy(functions = op.functions.map { fn ->
                val all = fn.statusByBackend["*"] ?: return@map fn
                val status = fn.statusByBackend.toMutableMap()
                status.remove("*")
                backends.forEach { backend -> status[backend] = mergeStatus(status[backend], all) }
                fn.copy(statusByBackend = status)
            })
        })
    }

    /** Status precedence: not_implemented > in_progress > implemented > inherited. */
    private fun mergeStatus(current: String?, incoming: String): String {
        val rank = listOf("not_implemented", "in_progress", "implemented", "inherited")
        if (current == null) return incoming
        val a = rank.indexOf(current).let { if (it < 0) rank.size else it }
        val b = rank.indexOf(incoming).let { if (it < 0) rank.size else it }
        return if (b < a) incoming else current
    }

    private fun readHloCoverage(): Set<String>? {
        val file = hloCoverageFile.orNull?.asFile?.takeIf { it.isFile } ?: return null
        return Json.decodeFromString<List<String>>(file.readText()).map { it.lowercase() }.toSet()
    }

    private fun readBackwardCoverage(): Map<String, String>? {
        val file = backwardCoverageFile.orNull?.asFile?.takeIf { it.isFile } ?: return null
        return Json.decodeFromString<Map<String, String>>(file.readText())
    }

    /** Render an issue reference (`#123`, a URL, or free text) as an AsciiDoc link. */
    private fun issueLink(issue: String): String {
        val trimmed = issue.trim()
        return when {
            trimmed.startsWith("http") -> "$trimmed[${trimmed.substringAfterLast('/').let { if (it.all(Char::isDigit)) "#$it" else "issue" }}]"
            trimmed.startsWith("#") && trimmed.drop(1).all(Char::isDigit) ->
                "${issueBaseUrl.getOrElse("https://github.com/SKaiNET-developers/SKaiNET/issues/")}${trimmed.drop(1)}[$trimmed]"
            else -> trimmed
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
            .filter { it != "*" }
            .toSortedSet()
            .toList()
        val hloOps = readHloCoverage()
        val backward = readBackwardCoverage()

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
            appendLine("Rows are `Operator.function` pairs. A gap is marked in source with `@NotImplemented(reason = …, issue = …)` (or `@InProgress`) and shows up here as ❌ (or 🚧) linked to its tracking issue; the PR that closes the gap removes the annotation. See xref:contributing/issue-taxonomy.adoc[Issue taxonomy] to pick one up.")
            appendLine("")
            appendLine("* *Validated* — whether the function's documentation has been DARC-validated by a reviewer (see xref:contributing/darc-workflow.adoc[DARC workflow]).")
            appendLine("* *Lowering* — the compile-path forms the function declares with `@Lowering` (`composite` = a `stablehlo.composite` when a target keeps it fused, `decompose` = StableHLO/CHLO primitives, `native` = a 1:1 target op); `—` means undeclared.")
            if (hloOps != null) appendLine("* *StableHLO* — ✅ the StableHLO converter can lower the op; ❌ export fails with \"No converter found\".")
            if (backward != null) appendLine("* *Backward* — for `@Diff` ops: ✅ a backward rule is wired; ❌ the rule is missing or a stub returning null gradients; `—` the op is not differentiable.")
            appendLine("* *Backend columns* — ✅ the backend overrides the function; ↪ the backend relies on the `TensorOps` default; 🚧 in progress; ❌ not implemented; `—` the backend makes no claim (unknown, not \"unsupported\").")
            appendLine("")
            if (rows.isEmpty() || allBackends.isEmpty()) {
                appendLine("NOTE: No backend status information found in the source data.")
                appendLine("")
                return@buildString
            }

            // Header: row label, validation column, then one column per backend.
            val extraCols = listOfNotNull(hloOps?.let { "StableHLO" }, backward?.let { "Backward" })
            val colSpec = (listOf("2", "1", "1") + List(extraCols.size + allBackends.size) { "1" }).joinToString(",")
            appendLine("[cols=\"$colSpec\", options=\"header\"]")
            appendLine("|===")
            append("| Operator.function | Validated | Lowering ")
            extraCols.forEach { append("| $it ") }
            allBackends.forEach { append("| $it ") }
            appendLine("")
            appendLine("")

            rows.forEach { row ->
                append("| `${row.operator}.${row.function.name}` ")
                append("| ${darcCell(row.function, row.hasPartial)} ")
                append("| ${row.function.lowering.ifEmpty { "—" }} ")
                if (hloOps != null) append("| ${if (row.function.name.lowercase() in hloOps) "✅" else "❌"} ")
                if (backward != null) append("| ${backwardCell(row.function, backward)} ")
                allBackends.forEach { backend ->
                    val raw = row.function.statusByBackend[backend]
                    val cell = if (raw == null) "—" else shortStatus(raw)
                    val issue = row.function.notes.firstOrNull { it.type == "issue" && (it.backend == backend || it.backend == "*") }
                    val link = if (issue != null && raw != null && !isDone(raw)) " ${issueLink(issue.message)}" else ""
                    append("| $cell$link ")
                }
                appendLine("")
            }

            // Totals footer: validated count, declared-lowering count, then per-backend done count.
            appendLine("")
            val validatedCount = rows.count { it.function.validated }
            val loweringCount = rows.count { it.function.lowering.isNotEmpty() }
            append("| *Done* | *$validatedCount / ${rows.size}* | *$loweringCount / ${rows.size}* ")
            if (hloOps != null) append("| *${rows.count { it.function.name.lowercase() in hloOps }} / ${rows.size}* ")
            if (backward != null) {
                val diffRows = rows.filter { it.function.isDifferentiable }
                append("| *${diffRows.count { backward[it.function.name] == "implemented" }} / ${diffRows.size}* ")
            }
            allBackends.forEach { backend ->
                val n = rows.count { isDone(it.function.statusByBackend[backend]) }
                append("| *$n / ${rows.size}* ")
            }
            appendLine("")
            appendLine("|===")
            appendLine("")
            appendLine("Per-function detail including notes lives in xref:reference/operators/generated/index.adoc[Operator reference].")
            appendLayersTable(module.layers)
        })
    }

    private fun backwardCell(function: FunctionDoc, backward: Map<String, String>): String {
        if (!function.isDifferentiable) return "—"
        return when (backward[function.name]) {
            "implemented" -> "✅"
            "stubbed" -> "❌ stub"
            else -> "❌"
        }
    }

    /** `nn` layer table: one row per concrete `Module` subclass, ❌ when `@NotImplemented`. */
    private fun StringBuilder.appendLayersTable(layers: List<LayerDoc>) {
        if (layers.isEmpty()) return
        appendLine("")
        appendLine("== NN layers")
        appendLine("")
        appendLine("Every concrete `sk.ainet.lang.nn` `Module` subclass. ❌ means the layer's `forward` throws today; the class carries `@NotImplemented`.")
        appendLine("")
        appendLine("[cols=\"2,3,1,3\", options=\"header\"]")
        appendLine("|===")
        appendLine("| Layer | Package | Status | Missing")
        appendLine("")
        layers.forEach { layer ->
            val missing = listOf(layer.reason, layer.issue.takeIf { it.isNotBlank() }?.let { issueLink(it) }.orEmpty())
                .filter { it.isNotBlank() }
                .joinToString(" — ")
                .ifEmpty { "—" }
            appendLine("| `${layer.name}` | `${layer.packageName}` | ${shortStatus(layer.status)} | $missing")
        }
        appendLine("")
        val done = layers.count { isDone(it.status) }
        appendLine("| *Done* | | *$done / ${layers.size}* |")
        appendLine("|===")
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
        "not_supported", "not_implemented", "missing", "unsupported" -> "❌"
        "inherited" -> "↪"
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
    
    /**
     * Markdown counterpart of [generateAsciidoc]: `index.md`, one `<operator>.md` page per
     * operator, and `ops-status-matrix.md` one level above, with the same rows and columns
     * as the AsciiDoc matrix. Antora `partial$` includes have no Markdown equivalent, so the
     * per-function prose is not emitted here.
     */
    private fun generateMarkdown(module: OperatorDocModule, outputDir: File) {
        outputDir.mkdirs()

        if (generateIndex.getOrElse(true)) {
            File(outputDir, "index.md").writeText(buildString {
                appendLine("# AI-NET Operators Reference")
                appendLine("")
                appendLine("Generated from version `${module.version}` on ${formatTimestamp(module.timestamp)}")
                appendLine("")
                appendLine("## Operators by Modality")
                appendLine("")
                module.operators.groupBy { it.modality }.forEach { (modality, operators) ->
                    appendLine("### ${modality.capitalize()}")
                    appendLine("")
                    operators.forEach { operator ->
                        appendLine("* [${operator.name}](${operator.name.lowercase()}.md)")
                    }
                    appendLine("")
                }
            })
        }

        module.operators.forEach { operator ->
            File(outputDir, "${operator.name.lowercase()}.md").writeText(buildString {
                appendLine("# ${operator.name}")
                appendLine("")
                appendLine("Package: `${operator.packageName}`")
                appendLine("")
                appendLine("Modality: ${operator.modality.capitalize()}")
                appendLine("")
                operator.functions.forEach { function -> appendMarkdownFunction(function, this) }
            })
        }

        if (includeBackendStatus.getOrElse(true)) {
            emitMarkdownStatusMatrix(module, outputDir)
        }
    }

    private fun appendMarkdownFunction(function: FunctionDoc, builder: StringBuilder) {
        builder.apply {
            appendLine("## ${function.name}")
            appendLine("")
            appendLine(if (function.validated) {
                val on = function.validatedOn.ifBlank { "an unspecified date" }
                val by = function.validatedBy.ifBlank { "an unspecified reviewer" }
                "**✅ DARC-validated by $by on $on**"
            } else {
                "**✖ Generated facts only (no human prose)**"
            })
            appendLine("")
            appendLine("### Signature")
            appendLine("")
            appendLine("```kotlin")
            appendLine(function.signature)
            appendLine("```")
            appendLine("")

            if (function.parameters.isNotEmpty()) {
                appendLine("### Parameters")
                appendLine("")
                function.parameters.forEach { param ->
                    appendLine("* `${param.name}: ${param.type}`")
                    if (param.description.isNotEmpty()) {
                        appendLine("  ${param.description}")
                    }
                }
                appendLine("")
            }

            appendLine("### Return Type")
            appendLine("")
            appendLine("`${function.returnType}`")
            appendLine("")

            if (function.notes.isNotEmpty()) {
                appendLine("### Notes")
                appendLine("")
                function.notes.forEach { note ->
                    val who = if (note.backend == "*") "all backends" else note.backend
                    val text = if (note.type == "issue") issueLinkMarkdown(note.message) else note.message
                    appendLine("> **$who**: $text")
                    appendLine("")
                }
            }
        }
    }

    /** Render an issue reference (`#123`, a URL, or free text) as a Markdown link. */
    private fun issueLinkMarkdown(issue: String): String {
        val trimmed = issue.trim()
        return when {
            trimmed.startsWith("http") -> "[${trimmed.substringAfterLast('/').let { if (it.all(Char::isDigit)) "#$it" else "issue" }}]($trimmed)"
            trimmed.startsWith("#") && trimmed.drop(1).all(Char::isDigit) ->
                "[$trimmed](${issueBaseUrl.getOrElse("https://github.com/SKaiNET-developers/SKaiNET/issues/")}${trimmed.drop(1)})"
            else -> trimmed
        }
    }

    /** Markdown twin of [emitOpsStatusMatrix]: same rows, columns and totals, as a GFM table. */
    private fun emitMarkdownStatusMatrix(module: OperatorDocModule, outputDir: File) {
        val matrixDir = outputDir.parentFile?.parentFile ?: outputDir
        matrixDir.mkdirs()
        val matrixFile = File(matrixDir, "ops-status-matrix.md")

        val allBackends: List<String> = module.operators
            .flatMap { op -> op.functions.flatMap { it.statusByBackend.keys } }
            .filter { it != "*" }
            .toSortedSet()
            .toList()
        val hloOps = readHloCoverage()
        val backward = readBackwardCoverage()

        val partialsRoot = derivePartialsRoot(outputDir)
        data class Row(val operator: String, val function: FunctionDoc, val hasPartial: Boolean)
        val rows: List<Row> = module.operators.flatMap { op ->
            op.functions.map { fn ->
                val relative = "ops/${op.name.lowercase()}/${fn.name.lowercase()}.adoc"
                val present = partialsRoot?.let { File(it, relative).isFile } == true
                Row(op.name, fn, present)
            }
        }

        matrixFile.writeText(buildString {
            appendLine("# Operator Coverage Matrix")
            appendLine("")
            appendLine("Generated from `operators.json` version `${module.version}` on ${formatTimestamp(module.timestamp)}.")
            appendLine("")
            appendLine("Rows are `Operator.function` pairs. A gap is marked in source with `@NotImplemented(reason = …, issue = …)` (or `@InProgress`) and shows up here as ❌ (or 🚧) linked to its tracking issue; the PR that closes the gap removes the annotation.")
            appendLine("")
            appendLine("* **Validated** — whether the function's documentation has been DARC-validated by a reviewer.")
            appendLine("* **Lowering** — the compile-path forms the function declares with `@Lowering` (`composite`, `decompose`, `native`); `—` means undeclared.")
            if (hloOps != null) appendLine("* **StableHLO** — ✅ the StableHLO converter can lower the op; ❌ export fails with \"No converter found\".")
            if (backward != null) appendLine("* **Backward** — for `@Diff` ops: ✅ a backward rule is wired; ❌ the rule is missing or a stub returning null gradients; `—` the op is not differentiable.")
            appendLine("* **Backend columns** — ✅ the backend overrides the function; ↪ the backend relies on the `TensorOps` default; 🚧 in progress; ❌ not implemented; `—` the backend makes no claim (unknown, not \"unsupported\").")
            appendLine("")
            if (rows.isEmpty() || allBackends.isEmpty()) {
                appendLine("NOTE: No backend status information found in the source data.")
                appendLine("")
                return@buildString
            }

            val extraCols = listOfNotNull(hloOps?.let { "StableHLO" }, backward?.let { "Backward" })
            append("| Operator.function | Validated | Lowering ")
            extraCols.forEach { append("| $it ") }
            allBackends.forEach { append("| $it ") }
            appendLine("|")
            append("| --- | --- | --- ")
            repeat(extraCols.size + allBackends.size) { append("| --- ") }
            appendLine("|")

            rows.forEach { row ->
                append("| `${row.operator}.${row.function.name}` ")
                append("| ${darcCell(row.function, row.hasPartial)} ")
                append("| ${row.function.lowering.ifEmpty { "—" }} ")
                if (hloOps != null) append("| ${if (row.function.name.lowercase() in hloOps) "✅" else "❌"} ")
                if (backward != null) append("| ${backwardCell(row.function, backward)} ")
                allBackends.forEach { backend ->
                    val raw = row.function.statusByBackend[backend]
                    val cell = if (raw == null) "—" else shortStatus(raw)
                    val issue = row.function.notes.firstOrNull { it.type == "issue" && (it.backend == backend || it.backend == "*") }
                    val link = if (issue != null && raw != null && !isDone(raw)) " ${issueLinkMarkdown(issue.message)}" else ""
                    append("| $cell$link ")
                }
                appendLine("|")
            }

            val validatedCount = rows.count { it.function.validated }
            val loweringCount = rows.count { it.function.lowering.isNotEmpty() }
            append("| **Done** | **$validatedCount / ${rows.size}** | **$loweringCount / ${rows.size}** ")
            if (hloOps != null) append("| **${rows.count { it.function.name.lowercase() in hloOps }} / ${rows.size}** ")
            if (backward != null) {
                val diffRows = rows.filter { it.function.isDifferentiable }
                append("| **${diffRows.count { backward[it.function.name] == "implemented" }} / ${diffRows.size}** ")
            }
            allBackends.forEach { backend ->
                val n = rows.count { isDone(it.function.statusByBackend[backend]) }
                append("| **$n / ${rows.size}** ")
            }
            appendLine("|")
            appendLine("")
            appendLine("Per-function detail including notes lives in [Operator reference](${outputDir.name}/index.md).")
            appendMarkdownLayersTable(module.layers)
        })
    }

    /** Markdown `nn` layer table, one row per concrete `Module` subclass. */
    private fun StringBuilder.appendMarkdownLayersTable(layers: List<LayerDoc>) {
        if (layers.isEmpty()) return
        appendLine("")
        appendLine("## NN layers")
        appendLine("")
        appendLine("Every concrete `sk.ainet.lang.nn` `Module` subclass. ❌ means the layer's `forward` throws today; the class carries `@NotImplemented`.")
        appendLine("")
        appendLine("| Layer | Package | Status | Missing |")
        appendLine("| --- | --- | --- | --- |")
        layers.forEach { layer ->
            val missing = listOf(layer.reason, layer.issue.takeIf { it.isNotBlank() }?.let { issueLinkMarkdown(it) }.orEmpty())
                .filter { it.isNotBlank() }
                .joinToString(" — ")
                .ifEmpty { "—" }
            appendLine("| `${layer.name}` | `${layer.packageName}` | ${shortStatus(layer.status)} | $missing |")
        }
        appendLine("")
        val done = layers.count { isDone(it.status) }
        appendLine("| **Done** | | **$done / ${layers.size}** | |")
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
                    val who = if (note.backend == "*") "all backends" else note.backend
                    val text = if (note.type == "issue") issueLink(note.message) else note.message
                    appendLine("TIP: *$who*: $text")
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