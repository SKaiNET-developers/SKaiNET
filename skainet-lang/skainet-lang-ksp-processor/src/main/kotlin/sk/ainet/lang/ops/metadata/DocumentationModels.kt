package sk.ainet.lang.ops.metadata

import kotlinx.serialization.Serializable

/**
 * Root documentation module containing all operator documentation for a module.
 */
@Serializable
data class OperatorDocModule(
    val schema: String = "https://skainet.ai/schemas/operator-doc/v1",
    val version: String,
    val commit: String,
    val timestamp: String,
    val module: String,
    val operators: List<OperatorDoc>
)

/**
 * Documentation for a single operator class.
 */
@Serializable
data class OperatorDoc(
    val name: String,
    val packageName: String,
    val modality: String,
    val functions: List<FunctionDoc>
)

/**
 * Documentation for a single function within an operator.
 */
@Serializable
data class FunctionDoc(
    val name: String,
    val signature: String,
    val parameters: List<ParameterDoc>,
    val returnType: String,
    val statusByBackend: Map<String, String>,
    val notes: List<Note>,
    val validated: Boolean = false,
    val validatedBy: String = "",
    val validatedOn: String = "",
    val validatedCommit: String = "",
    val referencesChecked: Boolean = true,
)

/**
 * Documentation for a function parameter.
 */
@Serializable
data class ParameterDoc(
    val name: String,
    val type: String,
    val description: String = ""
)

/**
 * A note associated with a function, typically containing owner or issue information.
 */
@Serializable
data class Note(
    val type: String, // "owner" or "issue"
    val backend: String,
    val content: String
)