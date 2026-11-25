package sk.ainet.buildlogic.json

import io.github.optimumcode.json.schema.JsonSchema
import io.github.optimumcode.json.schema.ValidationError
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import java.io.File
import java.io.InputStream

/**
 * Simple JSON Schema validator using OptimumCode/json-schema-validator.
 *
 * Usage example:
 *   val validator = SimpleJsonValidator()
 *   val result = validator.validate(schemaInputStream, jsonFile)
 */
class SimpleJsonValidator {

    private val json = Json { ignoreUnknownKeys = false }

    /**
     * Validate a JSON instance against a schema provided as InputStream.
     *
     * @param schemaStream InputStream of JSON Schema (draft-07 / 2019-09 / 2020-12)
     * @param instanceFile File containing the JSON data to validate
     * @param outputMode unused, kept for source compatibility
     * @return ValidationResult (valid flag + list of error messages)
     */
    fun validate(schemaStream: InputStream, instanceFile: File, outputMode: String = "basic"): ValidationResult {
        val schemaText = schemaStream.reader().use { it.readText() }
        val instanceText = instanceFile.readText()

        val schema = JsonSchema.fromDefinition(schemaText)
        val instance: JsonElement = json.parseToJsonElement(instanceText)

        val errors = mutableListOf<String>()
        val valid = schema.validate(instance) { error: ValidationError ->
            errors += "${error.objectPath}: ${error.message}"
        }
        // An invalid document always carries at least one message, so a caller that only
        // looks at the messages cannot take it for a valid one.
        if (!valid && errors.isEmpty()) errors += "document does not match the schema"
        return ValidationResult(valid, errors)
    }

    /**
     * Deprecated: use validate(schemaStream, instanceFile, outputMode) instead.
     */
    @Deprecated("Use validate(InputStream, File, String) instead")
    fun validate(schemaFile: File, instanceFile: File, outputMode: String = "basic"): ValidationResult =
        schemaFile.inputStream().use { validate(it, instanceFile, outputMode) }

    data class ValidationResult(
        val valid: Boolean,
        val errors: List<String>
    )
}
