package models

import kotlinx.serialization.Serializable

@Serializable
data class OperatorDocModule(
    val schema: String = "https://skainet.ai/schemas/operator-doc/v1",
    val version: String,
    val commit: String,
    val timestamp: String,
    val module: String,
    val operators: List<OperatorDoc>
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
    val validated: Boolean = false,
    val validatedBy: String = "",
    val validatedOn: String = "",
    val validatedCommit: String = "",
    val referencesChecked: Boolean = true,
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
    val message: String = ""
)

enum class DocumentationFormat {
    ASCIIDOC, MARKDOWN, HTML
}