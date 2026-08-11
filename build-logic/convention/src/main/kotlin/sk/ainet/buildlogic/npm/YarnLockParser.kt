package sk.ainet.buildlogic.npm

/**
 * Minimal reader for Yarn v1 lockfiles under `kotlin-js-store`.
 *
 * The format is a flat sequence of blocks:
 *
 * ```
 * ws@8.20.1, ws@~8.21.0:
 *   version "8.21.0"
 *   resolved "https://registry.yarnpkg.com/ws/-/ws-8.21.0.tgz#012e413f"
 * ```
 *
 * A block header is a non-indented line ending in `:`; the requested specs are
 * comma-separated and optionally quoted. Only the package name and the resolved
 * version are of interest here, so no attempt is made at a general TOML/YAML-ish
 * parse — anything that is neither a header nor a `version "…"` line is skipped.
 *
 * Deliberately free of Gradle types so it can be exercised by a plain unit test
 * once `build-logic` grows a test source set.
 */
internal object YarnLockParser {

    /** One entry per (spec, resolved version) pair; a package may legitimately appear more than once. */
    data class Entry(val packageName: String, val version: String)

    fun parse(text: String): List<Entry> {
        val entries = mutableListOf<Entry>()
        var pendingNames: List<String> = emptyList()

        for (rawLine in text.lineSequence()) {
            if (rawLine.isBlank() || rawLine.startsWith("#")) continue

            val isHeader = !rawLine.first().isWhitespace() && rawLine.trimEnd().endsWith(":")
            if (isHeader) {
                pendingNames = rawLine.trimEnd().dropLast(1)
                    .split(",")
                    .map { it.trim().trim('"') }
                    .filter { it.isNotEmpty() }
                    .map(::packageNameOf)
                    .distinct()
                continue
            }

            if (pendingNames.isEmpty()) continue

            val version = versionOf(rawLine.trim()) ?: continue
            pendingNames.forEach { entries += Entry(it, version) }
            pendingNames = emptyList()
        }

        return entries
    }

    /**
     * `ws@8.20.1` -> `ws`, `@types/node@^8.5.12` -> `@types/node`.
     * Splitting on the *last* `@` keeps scoped package names intact.
     */
    private fun packageNameOf(spec: String): String {
        val separator = spec.lastIndexOf('@')
        return if (separator <= 0) spec else spec.substring(0, separator)
    }

    private fun versionOf(line: String): String? {
        if (!line.startsWith("version")) return null
        return line.removePrefix("version").trim().removeSurrounding("\"").takeIf { it.isNotEmpty() }
    }
}
