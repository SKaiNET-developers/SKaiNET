package sk.ainet.io.onnx

import onnx.ModelProto
import onnx.StringStringEntryProto

/**
 * Parsed model metadata from an ONNX file.
 *
 * This class extracts common metadata fields from ONNX ModelProto's
 * metadataProps in a structured, type-safe manner. Supports various
 * model types including object detectors (YOLO), classifiers, etc.
 *
 * Usage:
 * ```kotlin
 * val modelBytes = File("model.onnx").readBytes()
 * val modelProto = ModelProto.decodeFromByteArray(modelBytes)
 * val metadata = OnnxModelMetadata.from(modelProto)
 *
 * println("Classes: ${metadata.classNames?.size ?: "none"}")
 * metadata.classNames?.forEach { (id, name) -> println("$id: $name") }
 * ```
 */
public data class OnnxModelMetadata(
    /** IR version of the ONNX model */
    val irVersion: Long,

    /** Producer name (e.g., "pytorch", "ultralytics") */
    val producerName: String?,

    /** Producer version */
    val producerVersion: String?,

    /** Model version */
    val modelVersion: Long,

    /** Model domain */
    val domain: String?,

    /** Doc string / description */
    val docString: String?,

    /** Class names for classification/detection models (classId -> className) */
    val classNames: Map<Int, String>?,

    /** Number of classes (may differ from classNames.size if some IDs are missing) */
    val numClasses: Int?,

    /** Model description from metadata */
    val description: String?,

    /** Model author from metadata */
    val author: String?,

    /** License information from metadata */
    val license: String?,

    /** Task type (e.g., "detect", "classify", "segment") */
    val task: String?,

    /** All raw metadata key-value pairs */
    val rawMetadata: Map<String, String>
) {
    public companion object {
        /**
         * Extract metadata from a parsed ONNX ModelProto.
         *
         * @param model The parsed ONNX model
         * @return Structured metadata
         */
        public fun from(model: ModelProto): OnnxModelMetadata {
            val rawMetadata = model.metadataProps.associate { it.key to it.value }

            return OnnxModelMetadata(
                irVersion = model.irVersion,
                producerName = model.producerName.takeIf { it.isNotBlank() },
                producerVersion = model.producerVersion.takeIf { it.isNotBlank() },
                modelVersion = model.modelVersion,
                domain = model.domain.takeIf { it.isNotBlank() },
                docString = model.docString.takeIf { it.isNotBlank() },
                classNames = parseClassNames(rawMetadata),
                numClasses = rawMetadata["num_classes"]?.toIntOrNull()
                    ?: rawMetadata["nc"]?.toIntOrNull()
                    ?: parseClassNames(rawMetadata)?.size,
                description = rawMetadata["description"]
                    ?: rawMetadata["desc"],
                author = rawMetadata["author"],
                license = rawMetadata["license"],
                task = rawMetadata["task"],
                rawMetadata = rawMetadata
            )
        }

        /**
         * Extract metadata from raw metadata props list.
         *
         * @param metadataProps The list of key-value pairs from ONNX model
         * @return Structured metadata
         */
        public fun from(metadataProps: List<StringStringEntryProto>): OnnxModelMetadata {
            val rawMetadata = metadataProps.associate { it.key to it.value }

            return OnnxModelMetadata(
                irVersion = 0L,
                producerName = null,
                producerVersion = null,
                modelVersion = 0L,
                domain = null,
                docString = null,
                classNames = parseClassNames(rawMetadata),
                numClasses = rawMetadata["num_classes"]?.toIntOrNull()
                    ?: rawMetadata["nc"]?.toIntOrNull()
                    ?: parseClassNames(rawMetadata)?.size,
                description = rawMetadata["description"]
                    ?: rawMetadata["desc"],
                author = rawMetadata["author"],
                license = rawMetadata["license"],
                task = rawMetadata["task"],
                rawMetadata = rawMetadata
            )
        }

        /**
         * Parse class names from ONNX metadata.
         *
         * YOLO/ultralytics format stores names as a Python dict string:
         * `{0: 'person', 1: 'bicycle', 2: 'car', ...}`
         *
         * Also checks for plain comma-separated format:
         * `person,bicycle,car,...`
         *
         * @param metadata Raw metadata map
         * @return Map of classId to className, or null if not found
         */
        public fun parseClassNames(metadata: Map<String, String>): Map<Int, String>? {
            // Try "names" key (ultralytics/YOLO format)
            val namesValue = metadata["names"]
                ?: metadata["class_names"]
                ?: metadata["classes"]
                ?: return null

            return parseClassNamesString(namesValue)
        }

        /**
         * Parse class names from a string value.
         *
         * Supports formats:
         * - Python dict: `{0: 'person', 1: 'bicycle'}`
         * - Comma-separated: `person,bicycle,car`
         *
         * @param value The string value to parse
         * @return Map of classId to className, or null if parsing fails
         */
        public fun parseClassNamesString(value: String): Map<Int, String>? {
            val trimmed = value.trim()

            // Try Python dict format: {0: 'person', 1: 'bicycle'}
            if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
                return parsePythonDictFormat(trimmed)
            }

            // Try comma-separated format: person,bicycle,car
            if (trimmed.contains(",")) {
                return parseCommaSeparatedFormat(trimmed)
            }

            return null
        }

        /**
         * Parse Python dict format: `{0: 'person', 1: 'bicycle', ...}`
         */
        private fun parsePythonDictFormat(value: String): Map<Int, String>? {
            val trimmed = value.trim().removePrefix("{").removeSuffix("}")
            if (trimmed.isEmpty()) return null

            val result = mutableMapOf<Int, String>()
            val entries = trimmed.split(",")

            for (entry in entries) {
                val parts = entry.split(":")
                if (parts.size >= 2) {
                    val id = parts[0].trim().trim('\'', '"').toIntOrNull()
                    val name = parts.subList(1, parts.size)
                        .joinToString(":")
                        .trim()
                        .trim('\'', '"')

                    if (id != null && name.isNotEmpty()) {
                        result[id] = name
                    }
                }
            }

            return result.toList().sortedBy { it.first }.toMap().takeIf { it.isNotEmpty() }
        }

        /**
         * Parse comma-separated format: `person,bicycle,car`
         */
        private fun parseCommaSeparatedFormat(value: String): Map<Int, String>? {
            val names = value.split(",")
                .map { it.trim() }
                .filter { it.isNotEmpty() }

            if (names.isEmpty()) return null

            return names.mapIndexed { index, name -> index to name }.toMap()
        }
    }
}
