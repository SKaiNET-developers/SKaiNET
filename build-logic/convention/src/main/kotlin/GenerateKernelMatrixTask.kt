import kotlinx.serialization.json.Json
import models.KernelSupportModule
import org.gradle.api.DefaultTask
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.CacheableTask
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction

/**
 * Renders `kernel-support.json` (registry-introspected by `KernelSupportMatrixTest`) into an
 * Antora AsciiDoc matrix — the kernel-side counterpart of [GenerateDocumentationTask]'s
 * `ops-status-matrix.adoc`. Rows are weight formats; columns are KMP platforms; each cell is
 * the best provider that serves `FP32 × format` on that platform.
 *
 * Deliberately omits a generation timestamp so the committed `.adoc` only changes when the
 * actual provider coverage changes (keeps the docs-CI staleness diff meaningful).
 */
@CacheableTask
abstract class GenerateKernelMatrixTask : DefaultTask() {

    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val inputFile: RegularFileProperty

    @get:OutputFile
    abstract val outputFile: RegularFileProperty

    @TaskAction
    fun generate() {
        val json = Json { ignoreUnknownKeys = true }
        val module = json.decodeFromString<KernelSupportModule>(inputFile.get().asFile.readText())
        val out = outputFile.get().asFile
        out.parentFile?.mkdirs()

        out.writeText(buildString {
            appendLine("= Kernel × platform support matrix")
            appendLine(":description: Which compute-kernel provider serves each weight format on each KMP target.")
            appendLine("")
            appendLine(
                "Generated from `kernel-support.json` (version `${module.version}`) by " +
                    "`KernelSupportMatrixTest` — registry introspection of the registered " +
                    "`KernelProvider` implementations. Do not edit by hand; run " +
                    "`./gradlew generateKernelMatrix` to refresh.",
            )
            appendLine("")
            appendLine(
                "Each cell is the best (highest-priority) provider that serves " +
                    "`${module.inputDtype} × format` `${module.op}` on that platform: " +
                    "*native-ffm* (100) → *panama-vector* (50) → *scalar* (0). An empty cell " +
                    "(`—`) means no provider carries a kernel there (the format is dequant-to-FP32 only).",
            )
            appendLine("")
            if (module.formats.isEmpty() || module.platforms.isEmpty()) {
                appendLine("NOTE: No kernel-support data found in the source JSON.")
                return@buildString
            }

            val colSpec = (listOf("1") + List(module.platforms.size) { "1" }).joinToString(",")
            appendLine("[cols=\"$colSpec\", options=\"header\"]")
            appendLine("|===")
            append("| Weight format ")
            module.platforms.forEach { append("| $it ") }
            appendLine("")
            appendLine("")
            module.formats.forEach { fmt ->
                append("| `${fmt.name}` ")
                module.platforms.forEach { p -> append("| ${fmt.byPlatform[p] ?: "—"} ") }
                appendLine("")
            }
            appendLine("|===")
            appendLine("")
            appendLine(
                "See also the eager backends & kernels mindmap " +
                    "(`docs/eager-execution-backends-and-kernels.md`) for the narrative overview and gaps.",
            )
        })
    }
}
