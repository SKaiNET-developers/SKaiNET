package models

import kotlinx.serialization.Serializable

@Serializable
data class OperatorDocModule(
    val schema: String = "https://skainet.ai/schemas/operator-doc/v1",
    val version: String,
    val commit: String,
    val timestamp: String,
    val module: String,
    val operators: List<OperatorDoc>,
    /** nn `Module` subclasses with their implementation status. */
    val layers: List<LayerDoc> = emptyList(),
    /** `Metric` implementations with the DARC validation of their `compute()`. */
    val metrics: List<MetricDoc> = emptyList(),
)

@Serializable
data class MetricDoc(
    val name: String,
    val packageName: String,
    val validated: Boolean = false,
    val validatedBy: String = "",
    val validatedOn: String = "",
    val validatedCommit: String = "",
    val referencesChecked: Boolean = true,
)

@Serializable
data class LayerDoc(
    val name: String,
    val packageName: String,
    val status: String,
    val reason: String = "",
    val issue: String = "",
)

@Serializable
data class OperatorDoc(
    val name: String,
    @kotlinx.serialization.SerialName("package")
    @kotlinx.serialization.json.JsonNames("packageName")
    val packageName: String,
    val modality: String,
    val functions: List<FunctionDoc>
)

@Serializable
data class FunctionDoc(
    val name: String,
    val signature: String,
    val parameters: List<ParameterDoc> = emptyList(),
    val returnType: String,
    val statusByBackend: Map<String, String> = emptyMap(),
    val notes: List<Note> = emptyList(),
    /** Whether the op carries `@Diff`, i.e. must have a wired backward rule. */
    val isDifferentiable: Boolean = false,
    val validated: Boolean = false,
    val validatedBy: String = "",
    val validatedOn: String = "",
    val validatedCommit: String = "",
    val referencesChecked: Boolean = true,
    /** Lowering forms from `@Lowering`: `composite`, `decompose`, `native` joined by ` · `; empty = undeclared. */
    val lowering: String = "",
)

@Serializable
data class ParameterDoc(
    val name: String,
    val type: String,
    val description: String = ""
)

@Serializable
data class Note(
    val type: String = "",
    val backend: String = "",
    // The KSP processor emits the note text as `content`.
    @kotlinx.serialization.json.JsonNames("content")
    val message: String = ""
)

enum class DocumentationFormat {
    ASCIIDOC, MARKDOWN, HTML
}