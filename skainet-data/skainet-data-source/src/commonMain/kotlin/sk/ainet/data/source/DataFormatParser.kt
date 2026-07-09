package sk.ainet.data.source

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** Supported built-in raw dataset formats. */
public enum class DataFormat(public val extensions: Set<String>) {
    CSV(setOf("csv")),
    TSV(setOf("tsv")),
    JSON(setOf("json")),
    JSON_LINES(setOf("jsonl", "ndjson"));

    public companion object {
        public fun fromExtension(extension: String): DataFormat? {
            val normalized = extension.trim().lowercase().removePrefix(".")
            if (normalized.isBlank()) return null
            return entries.firstOrNull { format -> normalized in format.extensions }
        }

        public fun inferFromFilename(filename: String): DataFormat? {
            val normalized = filename
                .substringBefore('?')
                .substringBefore('#')
                .trimEnd('/')
                .substringAfterLast('/')
            val extension = normalized.substringAfterLast('.', missingDelimiterValue = "")
            return fromExtension(extension)
        }
    }
}

/** A simple schema inferred from raw stringly parsed data. */
public data class DataSchema(
    public val columns: List<String>
)

/** One parsed row from a raw tabular or JSON-lines dataset. */
public data class RawDataRow(
    public val values: Map<String, String>
)

/** Parsed raw dataset plus schema and lightweight provenance metadata. */
public data class RawDataset(
    public val rows: List<RawDataRow>,
    public val schema: DataSchema,
    public val metadata: Map<String, String> = emptyMap()
)

/** Parser contract for converting source text into a [RawDataset]. */
public interface DataFormatParser {
    public val format: DataFormat

    public fun parse(text: String): RawDataset
}

/** Registry for built-in and user-provided data format parsers. */
public class DataFormatParserRegistry(
    parsers: Iterable<DataFormatParser> = defaultDataFormatParsers()
) {
    private val parsersByFormat: MutableMap<DataFormat, DataFormatParser> =
        parsers.associateBy { it.format }.toMutableMap()

    public fun register(parser: DataFormatParser) {
        parsersByFormat[parser.format] = parser
    }

    public fun parserFor(format: DataFormat): DataFormatParser =
        parsersByFormat[format] ?: throw DataSourceException("No parser registered for $format")

    public fun parse(format: DataFormat, text: String): RawDataset =
        parserFor(format).parse(text)

    public companion object {
        public fun default(): DataFormatParserRegistry = DataFormatParserRegistry()
    }
}

/** Returns the default built-in parser set. */
public fun defaultDataFormatParsers(): List<DataFormatParser> =
    listOf(
        DelimitedTextDataFormatParser(DataFormat.CSV, delimiter = ','),
        DelimitedTextDataFormatParser(DataFormat.TSV, delimiter = '\t'),
        JsonDataFormatParser(),
        JsonLinesDataFormatParser()
    )

/** Parser for simple delimited text with a required header row. */
public class DelimitedTextDataFormatParser(
    override val format: DataFormat,
    private val delimiter: Char
) : DataFormatParser {
    override fun parse(text: String): RawDataset {
        val lines = text.lineSequence()
            .map { it.trimEnd('\r') }
            .filter { it.isNotBlank() }
            .toList()
        require(lines.isNotEmpty()) { "$format input must contain a header row" }

        val columns = splitDelimitedLine(lines.first(), delimiter)
        require(columns.all { it.isNotBlank() }) { "$format header columns must not be blank" }

        val rows = lines.drop(1).mapIndexed { index, line ->
            val values = splitDelimitedLine(line, delimiter)
            require(values.size == columns.size) {
                "$format row ${index + 2} has ${values.size} values but expected ${columns.size}"
            }
            RawDataRow(columns.zip(values).toMap())
        }

        return RawDataset(
            rows = rows,
            schema = DataSchema(columns),
            metadata = mapOf("format" to format.name, "rowCount" to rows.size.toString())
        )
    }
}

/** Parser for JSON object datasets encoded as one object or an array of objects. */
public class JsonDataFormatParser(
    private val json: Json = Json
) : DataFormatParser {
    override val format: DataFormat = DataFormat.JSON

    override fun parse(text: String): RawDataset {
        val root = json.parseToJsonElement(text)
        val objects = when (root) {
            is JsonObject -> listOf(root)
            is JsonArray -> root.mapIndexed { index, element ->
                require(element is JsonObject) { "JSON array element ${index + 1} must be an object" }
                element
            }
            else -> throw IllegalArgumentException("JSON input must be an object or array of objects")
        }

        return objects.toRawDataset(format)
    }
}

/** Parser for newline-delimited JSON objects. */
public class JsonLinesDataFormatParser(
    private val json: Json = Json
) : DataFormatParser {
    override val format: DataFormat = DataFormat.JSON_LINES

    override fun parse(text: String): RawDataset {
        val objects = text.lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .mapIndexed { index, line ->
                val element = json.parseToJsonElement(line)
                require(element is JsonObject) { "JSON_LINES row ${index + 1} must be a JSON object" }
                element
            }
            .toList()

        return objects.toRawDataset(format)
    }
}

private fun List<JsonObject>.toRawDataset(format: DataFormat): RawDataset {
    val columns = flatMap { it.keys }.distinct()
    val rows = map { obj ->
        RawDataRow(columns.associateWith { column -> obj[column]?.toRawString().orEmpty() })
    }

    return RawDataset(
        rows = rows,
        schema = DataSchema(columns),
        metadata = mapOf("format" to format.name, "rowCount" to rows.size.toString())
    )
}

private fun splitDelimitedLine(line: String, delimiter: Char): List<String> {
    val values = mutableListOf<String>()
    val current = StringBuilder()
    var quoted = false
    var index = 0

    while (index < line.length) {
        val char = line[index]
        when {
            char == '"' && quoted && index + 1 < line.length && line[index + 1] == '"' -> {
                current.append('"')
                index++
            }
            char == '"' -> quoted = !quoted
            char == delimiter && !quoted -> {
                values.add(current.toString())
                current.clear()
            }
            else -> current.append(char)
        }
        index++
    }

    require(!quoted) { "Unterminated quoted field" }
    values.add(current.toString())
    return values
}

private fun JsonElement.toRawString(): String =
    when (this) {
        JsonNull -> ""
        is JsonPrimitive -> content
        is JsonArray -> toString()
        is JsonObject -> toString()
    }
