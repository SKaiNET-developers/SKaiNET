package sk.ainet.lang.ops.ksp

import com.google.devtools.ksp.processing.*
import com.google.devtools.ksp.symbol.*
import com.google.devtools.ksp.validate
import java.time.Instant

// Simple data classes for documentation generation
data class OperatorDocModule(
    val schema: String = "https://skainet.ai/schemas/operator-doc/v1",
    val version: String,
    val commit: String,
    val timestamp: String,
    val module: String,
    val operators: List<OperatorDoc>
)

data class OperatorDoc(
    val name: String,
    val packageName: String,
    val modality: String,
    val functions: List<FunctionDoc>
)

data class FunctionDoc(
    val name: String,
    val signature: String,
    val parameters: List<ParameterDoc>,
    val returnType: String,
    val statusByBackend: Map<String, String>,
    val notes: List<Note>
)

data class ParameterDoc(
    val name: String,
    val type: String,
    val description: String = ""
)

data class Note(
    val type: String,
    val backend: String,
    val content: String
)

/**
 * KSP processor that generates operator documentation by scanning for functions and classes
 * annotated with @NotImplemented and @InProgress annotations, and creates JSON output
 * following the OperatorDocModule schema.
 */
class OperatorDocProcessor(
    private val codeGenerator: CodeGenerator,
    private val logger: KSPLogger
) : SymbolProcessor {

    override fun process(resolver: Resolver): List<KSAnnotated> {
        logger.info("Starting OperatorDocProcessor...")

        val notImplementedSymbols = resolver
            .getSymbolsWithAnnotation("sk.ainet.lang.ops.NotImplemented")
            .filterIsInstance<KSDeclaration>()
            .filter { it.validate() }

        val inProgressSymbols = resolver
            .getSymbolsWithAnnotation("sk.ainet.lang.ops.InProgress")
            .filterIsInstance<KSDeclaration>()
            .filter { it.validate() }
            
        val testInProgressSymbols = resolver
            .getSymbolsWithAnnotation("test.InProgress")
            .filterIsInstance<KSDeclaration>()
            .filter { it.validate() }

        val dslOpSymbols = resolver
            .getSymbolsWithAnnotation("sk.ainet.lang.ops.DslOp")
            .filterIsInstance<KSDeclaration>()
            .filter { it.validate() }

        val allSymbols = (notImplementedSymbols + inProgressSymbols + testInProgressSymbols + dslOpSymbols).toList()

        if (allSymbols.isEmpty()) {
            logger.info("No annotated symbols found")
            return emptyList()
        }

        logger.info("Found ${allSymbols.size} annotated symbols")

        // Group symbols by their containing class/package to create operators
        val operatorDocs = groupSymbolsByOperator(allSymbols)

        // Create the module documentation
        val module = OperatorDocModule(
            version = extractVersion(),
            commit = extractCommitSha(),
            timestamp = Instant.now().toString(),
            module = "skainet-lang-core", // TODO: Extract from module info
            operators = operatorDocs
        )

        // Generate JSON output
        generateJsonOutput(module)

        return emptyList() // No symbols need further processing
    }

    private fun groupSymbolsByOperator(symbols: List<KSDeclaration>): List<OperatorDoc> {
        return symbols
            .groupBy { symbol ->
                when (symbol) {
                    is KSFunctionDeclaration -> {
                        val parent = symbol.parentDeclaration
                        if (parent is KSClassDeclaration) {
                            parent
                        } else {
                            // If no parent class, use a virtual class based on category or package
                            null
                        }
                    }
                    is KSClassDeclaration -> symbol
                    else -> null
                }
            }
            .map { (classSymbol, declarations) ->
                if (classSymbol != null) {
                    createOperatorDoc(classSymbol, declarations)
                } else {
                    // Handle top-level functions (like DSL ops)
                    createVirtualOperatorDoc(declarations)
                }
            }
    }

    private fun createVirtualOperatorDoc(declarations: List<KSDeclaration>): OperatorDoc {
        val firstFunc = declarations.filterIsInstance<KSFunctionDeclaration>().firstOrNull()
        val packageName = firstFunc?.packageName?.asString() ?: "sk.ainet.lang.ops"
        
        val functions = declarations.filterIsInstance<KSFunctionDeclaration>()
            .map { createFunctionDoc(it) }

        // Use category from @DslOp if available for the operator name
        val name = firstFunc?.annotations?.find { it.shortName.asString() == "DslOp" }
            ?.arguments?.find { it.name?.asString() == "category" }?.value?.toString()
            ?.takeIf { it.isNotEmpty() } ?: "Composite"

        return OperatorDoc(
            name = name,
            packageName = packageName,
            modality = "composite",
            functions = functions
        )
    }

    private fun createOperatorDoc(classSymbol: KSClassDeclaration, declarations: List<KSDeclaration>): OperatorDoc {
        val functions = declarations.filterIsInstance<KSFunctionDeclaration>()
            .map { createFunctionDoc(it) }

        return OperatorDoc(
            name = classSymbol.simpleName.asString(),
            packageName = classSymbol.packageName.asString(),
            modality = extractModality(classSymbol),
            functions = functions
        )
    }

    private fun createFunctionDoc(function: KSFunctionDeclaration): FunctionDoc {
        return FunctionDoc(
            name = function.simpleName.asString(),
            signature = function.toSignatureString(),
            parameters = extractParameters(function),
            returnType = extractReturnType(function),
            statusByBackend = deriveStatusByBackend(function),
            notes = deriveNotes(function)
        )
    }

    private fun KSFunctionDeclaration.toSignatureString(): String {
        val params = parameters.joinToString(", ") { param ->
            "${param.name?.asString() ?: ""}:${param.type.resolve().declaration.simpleName.asString()}"
        }
        val returnType = returnType?.resolve()?.declaration?.simpleName?.asString() ?: "Unit"
        return "fun ${simpleName.asString()}($params): $returnType"
    }

    private fun extractParameters(function: KSFunctionDeclaration): List<ParameterDoc> {
        return function.parameters.map { param ->
            ParameterDoc(
                param.name?.asString() ?: "",
                param.type.resolve().declaration.simpleName.asString(),
                "" // TODO: Extract from KDoc if available
            )
        }
    }

    private fun extractReturnType(function: KSFunctionDeclaration): String {
        return function.returnType?.resolve()?.declaration?.simpleName?.asString() ?: "Unit"
    }

    private fun deriveStatusByBackend(declaration: KSDeclaration): Map<String, String> {
        val statusMap = mutableMapOf<String, String>()

        // Check @DslOp annotation - if present, it's implemented by definition as it's a composite op
        declaration.annotations.find {
            it.shortName.asString() == "DslOp"
        }?.let {
            statusMap["cpu"] = "implemented" // Assuming composite ops are implemented on all backends by default
            statusMap["wasm"] = "implemented"
            statusMap["apple"] = "implemented"
        }

        // Check @InProgress annotation
        declaration.annotations.find {
            it.shortName.asString() == "InProgress"
        }?.let { annotation ->
            logger.info("Processing annotation: ${annotation.shortName.asString()}")
            logger.info("Annotation arguments: ${annotation.arguments.map { "${it.name?.asString()}: ${it.value}" }}")

            // For vararg parameters, the first argument contains the array
            val backendsArg = annotation.arguments.firstOrNull()
            val backends = when (val value = backendsArg?.value) {
                is List<*> -> value.map { it.toString() }
                is String -> listOf(value)
                else -> emptyList()
            }
            backends.forEach { backend ->
                statusMap[backend] = "in_progress"
            }
        }
        return statusMap
    }

    private fun deriveNotes(declaration: KSDeclaration): List<Note> {
        val notes = mutableListOf<Note>()

        // Extract description from @DslOp
        declaration.annotations.find {
            it.shortName.asString() == "DslOp"
        }?.let { annotation ->
            val description = annotation.arguments.find { it.name?.asString() == "description" }
                ?.value?.toString() ?: ""
            if (description.isNotEmpty()) {
                notes.add(Note("description", "all", description))
            }
        }

        // Extract notes from @InProgress annotation
        declaration.annotations.find {
            it.shortName.asString() == "InProgress"
        }?.let { annotation ->
            // For vararg parameters, the first argument contains the array
            val backendsArg = annotation.arguments.firstOrNull()
            val backends = when (val value = backendsArg?.value) {
                is List<*> -> value.map { it.toString() }
                is String -> listOf(value)
                else -> emptyList()
            }
            val owner = annotation.arguments.find { it.name?.asString() == "owner" }
                ?.value?.toString() ?: ""
            val issue = annotation.arguments.find { it.name?.asString() == "issue" }
                ?.value?.toString() ?: ""

            backends.forEach { backend ->
                if (owner.isNotEmpty()) {
                    notes.add(Note("owner", backend, owner))
                }
                if (issue.isNotEmpty()) {
                    notes.add(Note("issue", backend, issue))
                }
            }
        }

        return notes
    }

    private fun extractModality(classSymbol: KSClassDeclaration): String {
        // Simple heuristic based on package or class name
        val packageName = classSymbol.packageName.asString()
        return when {
            packageName.contains("vision") -> "vision"
            packageName.contains("nlp") || packageName.contains("text") -> "nlp"
            packageName.contains("audio") -> "audio"
            else -> "core"
        }
    }

    private fun extractVersion(): String {
        // TODO: Extract from project metadata
        return "1.0.0"
    }

    private fun extractCommitSha(): String {
        // TODO: Extract from git metadata
        return "unknown"
    }

    private fun escapeJson(value: String): String = buildString {
        value.forEach { ch ->
            when (ch) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f") // form feed
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    if (ch < ' ') {
                        append("\\u")
                        append(ch.code.toString(16).padStart(4, '0'))
                    } else append(ch)
                }
            }
        }
    }

    private fun generateJsonOutput(module: OperatorDocModule) {
        try {
            // Simple JSON generation without external dependencies
            val jsonContent = buildString {
                append("{\n")
                append("  \"schema\": \"${escapeJson(module.schema)}\",\n")
                append("  \"version\": \"${escapeJson(module.version)}\",\n")
                append("  \"commit\": \"${escapeJson(module.commit)}\",\n")
                append("  \"timestamp\": \"${escapeJson(module.timestamp)}\",\n")
                append("  \"module\": \"${escapeJson(module.module)}\",\n")
                append("  \"operators\": [\n")

                module.operators.forEachIndexed { opIndex, operator ->
                    append("    {\n")
                    append("      \"name\": \"${escapeJson(operator.name)}\",\n")
                    append("      \"packageName\": \"${escapeJson(operator.packageName)}\",\n")
                    append("      \"modality\": \"${escapeJson(operator.modality)}\",\n")
                    append("      \"functions\": [\n")

                    operator.functions.forEachIndexed { funcIndex, function ->
                        append("        {\n")
                        append("          \"name\": \"${escapeJson(function.name)}\",\n")
                        append("          \"signature\": \"${escapeJson(function.signature)}\",\n")
                        // parameters
                        append("          \"parameters\": [")
                        function.parameters.forEachIndexed { pIndex, p ->
                            append("{\"name\": \"${escapeJson(p.name)}\", \"type\": \"${escapeJson(p.type)}\", \"description\": \"${escapeJson(p.description)}\"}")
                            if (pIndex < function.parameters.size - 1) append(", ")
                        }
                        append("],\n")
                        append("          \"returnType\": \"${escapeJson(function.returnType)}\",\n")

                        // Generate statusByBackend JSON
                        append("          \"statusByBackend\": {")
                        function.statusByBackend.entries.forEachIndexed { statusIndex, (backend, status) ->
                            append("\"${escapeJson(backend)}\": \"${escapeJson(status)}\"")
                            if (statusIndex < function.statusByBackend.size - 1) append(", ")
                        }
                        append("},\n")

                        // Generate notes JSON
                        append("          \"notes\": [")
                        function.notes.forEachIndexed { noteIndex, note ->
                            append("{\"type\": \"${escapeJson(note.type)}\", \"backend\": \"${escapeJson(note.backend)}\", \"content\": \"${escapeJson(note.content)}\"}")
                            if (noteIndex < function.notes.size - 1) append(", ")
                        }
                        append("]\n")

                        append("        }")
                        if (funcIndex < operator.functions.size - 1) append(",")
                        append("\n")
                    }

                    append("      ]\n")
                    append("    }")
                    if (opIndex < module.operators.size - 1) append(",")
                    append("\n")
                }

                append("  ]\n")
                append("}")
            }

            val file = codeGenerator.createNewFile(
                dependencies = Dependencies.ALL_FILES,
                packageName = "",
                fileName = "operators",
                extensionName = "json"
            )

            file.write(jsonContent.toByteArray())
            file.close()

            logger.info("Generated operators.json with ${module.operators.size} operators")
        } catch (e: Exception) {
            logger.error("Failed to generate JSON output: ${e.message}")
        }
    }
}

/**
 * Provider for the OperatorDocProcessor.
 */
class OperatorDocProcessorProvider : SymbolProcessorProvider {
    override fun create(environment: SymbolProcessorEnvironment): SymbolProcessor {
        return OperatorDocProcessor(environment.codeGenerator, environment.logger)
    }
}