import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.*
import java.io.File

/**
 * Guards a SKEEP-003 acceptance criterion: no dispatch site downcasts to a concrete `TensorData`
 * class outside the data package itself. Kernels are meant to be chosen from a view's
 * `Format(dtype, encoding)`, not from the Kotlin class that happens to hold the bytes.
 *
 * A downcast is an `is` / `!is` / `as` / `as?` whose target is a class named `…TensorData`
 * (the generic base `TensorData` itself does not count) or one of the storage markers in
 * [markerTypes]. Production code only; the packages that define the classes are exempt.
 *
 * The sites that exist today are listed in [allowlistFile] as `path|count|reason`, where `count`
 * is the exact number of downcasts in the file. A new downcast fails the build until the entry is
 * raised, which a reviewer sees; removing one fails until the entry is lowered, so the list can
 * only describe the code as it is. The goal is an empty list.
 */
@CacheableTask
abstract class CheckTensorDataDowncastsTask : DefaultTask() {

    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sources: ConfigurableFileCollection

    /** Repository root that report and allowlist paths are relative to. */
    @get:Internal
    abstract val rootDirectory: DirectoryProperty

    @get:InputFile
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val allowlistFile: RegularFileProperty

    @get:OutputFile
    abstract val reportFile: RegularFileProperty

    private val cast = Regex("""(?:!is|\bis|\bas\?|\bas)\s+(?:[A-Za-z_][\w]*\.)*([A-Z]\w*)""")
    private val mainSourceDir = Regex("""/src/[A-Za-z0-9]*[Mm]ain/""")

    /** Marker interfaces that stand for a concrete storage class in a dispatch `when`. */
    private val markerTypes = setOf(
        "PackedBlockStorage",
        "RowDequantSource",
        "MemorySegmentBackedData",
        "Q4MemorySegmentMarker",
        "Q8MemorySegmentMarker",
    )

    /** Packages that define the classes and may inspect them. */
    private val exemptPackages = listOf(
        "/sk/ainet/lang/tensor/data/",
        "/sk/ainet/lang/tensor/storage/",
        "/sk/ainet/lang/memory/",
    )

    @TaskAction
    fun check() {
        val root = rootDirectory.get().asFile
        val allowlist = parseAllowlist(allowlistFile.get().asFile)
        val found = mutableMapOf<String, MutableList<Int>>()

        sources.files
            .filter { it.extension == "kt" }
            .forEach { file ->
                val rel = file.relativeTo(root).invariantSeparatorsPath
                if (!mainSourceDir.containsMatchIn("/$rel")) return@forEach
                if (exemptPackages.any { it in "/$rel" }) return@forEach
                file.readLines().forEachIndexed { index, line ->
                    val trimmed = line.trimStart()
                    if (trimmed.startsWith("//") || trimmed.startsWith("*") || trimmed.startsWith("/*")) return@forEachIndexed
                    val code = stripStrings(line).substringBefore("//")
                    val hits = cast.findAll(code).count { isConcreteStorageType(it.groupValues[1]) }
                    repeat(hits) { found.getOrPut(rel) { mutableListOf() } += index + 1 }
                }
            }

        val problems = mutableListOf<String>()
        found.toSortedMap().forEach { (path, hits) ->
            val expected = allowlist[path]
            when {
                expected == null -> hits.forEach { problems += "$path:$it — downcast to a concrete TensorData class" }
                expected != hits.size -> problems += "$path — ${hits.size} downcasts, allowlist expects $expected (lines ${hits.joinToString()})"
            }
        }
        allowlist.keys.filter { it !in found }.sorted().forEach { path ->
            problems += "$path — allowlisted but has no downcasts any more; remove the entry"
        }

        reportFile.get().asFile.apply { parentFile.mkdirs() }.writeText(
            found.toSortedMap().entries.joinToString("\n", postfix = "\n") { (path, hits) -> "$path|${hits.size}|${hits.joinToString(",")}" },
        )

        if (problems.isNotEmpty()) {
            throw GradleException(
                buildString {
                    appendLine("Downcasts to concrete TensorData classes outside the data package (SKEEP-003):")
                    problems.forEach { appendLine("  $it") }
                    appendLine()
                    appendLine("Choose the kernel from the view's Format(dtype, encoding) instead of the class. If the")
                    appendLine("count went down, lower it in ${allowlistFile.get().asFile.relativeTo(root).invariantSeparatorsPath} (path|count|reason).")
                },
            )
        }
        logger.lifecycle("✅ TensorData downcasts match the allowlist (${found.values.sumOf { it.size }} in ${found.size} files)")
    }

    private fun isConcreteStorageType(name: String): Boolean =
        name in markerTypes || (name.endsWith("TensorData") && name != "TensorData")

    /** Blank out the contents of `"…"` literals on one line (escape-aware). */
    private fun stripStrings(line: String): String = buildString {
        var inString = false
        var i = 0
        while (i < line.length) {
            val ch = line[i]
            when {
                inString && ch == '\\' -> i++
                ch == '"' -> { inString = !inString; append(ch) }
                !inString -> append(ch)
            }
            i++
        }
    }

    private fun parseAllowlist(file: File): Map<String, Int> =
        file.readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") }
            .associate { line ->
                val parts = line.split("|").map { it.trim() }
                if (parts.size < 3 || parts[2].isEmpty()) {
                    throw GradleException("Malformed allowlist line (want path|count|reason): $line")
                }
                parts[0] to (parts[1].toIntOrNull() ?: throw GradleException("Bad count in allowlist line: $line"))
            }
}
