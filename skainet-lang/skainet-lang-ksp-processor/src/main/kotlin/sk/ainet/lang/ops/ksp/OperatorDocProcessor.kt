package sk.ainet.lang.ops.ksp

import com.google.devtools.ksp.getClassDeclarationByName
import com.google.devtools.ksp.getDeclaredFunctions
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
    val notes: List<Note>,
    // DARC validation metadata. `validated = false` means @DarcValidated is
    // absent — the generator will render a "not validated" badge.
    val validated: Boolean = false,
    val validatedBy: String = "",
    val validatedOn: String = "",
    val validatedCommit: String = "",
    val referencesChecked: Boolean = true,
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
    private val logger: KSPLogger,
    private val options: Map<String, String> = emptyMap(),
) : SymbolProcessor {

    private var alreadyGenerated = false

    override fun process(resolver: Resolver): List<KSAnnotated> {
        if (alreadyGenerated) return emptyList()
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

        val rawSymbols = (notImplementedSymbols + inProgressSymbols + testInProgressSymbols + dslOpSymbols).toList()

        // Drop symbols whose enclosing class is `@Backend`-tagged: those are
        // backend implementors of `TensorOps` and their coverage is already
        // reflected in the TensorOps surface scan's backend matrix. Emitting a
        // standalone page for them would duplicate info and, worse, produce a
        // stub page showing only the handful of methods that happen to carry
        // a status annotation.
        val allSymbols = rawSymbols.filterNot { symbol ->
            val parent = (symbol as? KSFunctionDeclaration)?.parentDeclaration as? KSClassDeclaration
            parent?.annotations?.any { it.shortName.asString() == "Backend" } == true
        }

        logger.info("Found ${allSymbols.size} annotated symbols (dropped ${rawSymbols.size - allSymbols.size} on @Backend classes)")

        // Group annotation-discovered symbols by their containing class/package to create operators
        val annotationOps = if (allSymbols.isNotEmpty()) groupSymbolsByOperator(allSymbols) else emptyList()

        // Additionally discover the full TensorOps surface by walking the interface
        // and any `@Backend`-tagged implementors visible in this compilation unit.
        // This scales coverage beyond the hand-annotated symbols and makes the
        // backend matrix track ground truth instead of annotation drift.
        val interfaceOps = discoverTensorOpsSurface(resolver)

        // Prefer interface-scan for the TensorOps operator; keep annotation-derived
        // operators (like synthetic Similarity from @DslOp) untouched.
        val interfaceNames = interfaceOps.map { it.name }.toSet()
        val operatorDocs = interfaceOps + annotationOps.filter { it.name !in interfaceNames }

        if (operatorDocs.isEmpty()) {
            logger.info("No operators discovered (no annotations and no TensorOps visible)")
            return emptyList()
        }

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
        alreadyGenerated = true

        return emptyList() // No symbols need further processing
    }

    /**
     * Walk the `TensorOps` interface and every `@Backend`-annotated class
     * visible in this compilation unit to produce a single `OperatorDoc`
     * covering the full op surface. Each function's `statusByBackend` maps
     * every visible backend id to `implemented` when that backend's class
     * declares an override of the method, or `inherited` otherwise.
     *
     * Returns an empty list when `TensorOps` is not on the compilation
     * classpath — non-`skainet-lang-core` modules simply fall back to the
     * annotation-driven path.
     */
    private fun discoverTensorOpsSurface(resolver: Resolver): List<OperatorDoc> {
        val tensorOpsName = resolver.getKSNameFromString("sk.ainet.lang.tensor.ops.TensorOps")
        val tensorOps = resolver.getClassDeclarationByName(tensorOpsName) ?: return emptyList()

        // Backend classes marked `internal = true` are shape/dtype
        // sentinels or test doubles (e.g. `VoidTensorOps`). Drop them
        // from the surface scan so they never appear in user-facing
        // pages or coverage matrices.
        val backendClasses: List<Pair<String, KSClassDeclaration>> = resolver
            .getSymbolsWithAnnotation("sk.ainet.lang.ops.Backend")
            .filterIsInstance<KSClassDeclaration>()
            .mapNotNull { cls ->
                val ann = cls.annotations.find { it.shortName.asString() == "Backend" } ?: return@mapNotNull null
                val isInternal = ann.arguments.find { it.name?.asString() == "internal" }?.value as? Boolean == true
                if (isInternal) return@mapNotNull null
                val id = ann.arguments.find { it.name?.asString() == "id" }?.value?.toString()
                    ?: return@mapNotNull null
                id to cls
            }
            .toList()

        logger.info("Discovered ${backendClasses.size} @Backend classes for TensorOps surface scan")

        // Interface methods only — skip default implementations authored on
        // the interface (they still show up here but their status defaults
        // to "inherited" for every backend unless overridden).
        val interfaceFunctions = tensorOps.getDeclaredFunctions().toList()

        val functionDocs = interfaceFunctions.map { fn ->
            val statusByBackend = mutableMapOf<String, String>()
            for ((backendId, backendClass) in backendClasses) {
                val overrides = backendClass.getAllFunctions().any { candidate ->
                    candidate.simpleName.asString() == fn.simpleName.asString() &&
                        candidate.findOverridee()?.simpleName?.asString() == fn.simpleName.asString()
                }
                statusByBackend[backendId] = if (overrides) "implemented" else "inherited"
            }
            val validation = extractDarcValidation(fn)
            FunctionDoc(
                name = fn.simpleName.asString(),
                signature = fn.toSignatureString(),
                parameters = extractParameters(fn),
                returnType = extractReturnType(fn),
                statusByBackend = statusByBackend,
                notes = emptyList(),
                validated = validation.validated,
                validatedBy = validation.by,
                validatedOn = validation.on,
                validatedCommit = validation.commit,
                referencesChecked = validation.referencesChecked,
            )
        }

        return listOf(
            OperatorDoc(
                name = "TensorOps",
                packageName = tensorOps.packageName.asString(),
                modality = "core",
                functions = functionDocs
            )
        )
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
        val validation = extractDarcValidation(function)
        return FunctionDoc(
            name = function.simpleName.asString(),
            signature = function.toSignatureString(),
            parameters = extractParameters(function),
            returnType = extractReturnType(function),
            statusByBackend = deriveStatusByBackend(function),
            notes = deriveNotes(function),
            validated = validation.validated,
            validatedBy = validation.by,
            validatedOn = validation.on,
            validatedCommit = validation.commit,
            referencesChecked = validation.referencesChecked,
        )
    }

    private data class DarcValidation(
        val validated: Boolean,
        val by: String,
        val on: String,
        val commit: String,
        val referencesChecked: Boolean,
    )

    /**
     * Read the `@DarcValidated` annotation off a function, if present.
     * Returns a sentinel with `validated = false` when the annotation is
     * absent, which the generator renders as the "not validated" badge.
     */
    private fun extractDarcValidation(function: KSFunctionDeclaration): DarcValidation {
        val annotation = function.annotations.find {
            it.shortName.asString() == "DarcValidated"
        } ?: return DarcValidation(false, "", "", "", true)

        val by = annotation.arguments.find { it.name?.asString() == "by" }
            ?.value?.toString().orEmpty()
        val on = annotation.arguments.find { it.name?.asString() == "on" }
            ?.value?.toString().orEmpty()
        val commit = annotation.arguments.find { it.name?.asString() == "commit" }
            ?.value?.toString().orEmpty()
        val refsChecked = (annotation.arguments.find {
            it.name?.asString() == "referencesChecked"
        }?.value as? Boolean) ?: true

        return DarcValidation(
            validated = true,
            by = by,
            on = on,
            commit = commit,
            referencesChecked = refsChecked,
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
        val paramDocs = parseKDocParams(function.docString)
        return function.parameters.map { param ->
            val name = param.name?.asString() ?: ""
            ParameterDoc(
                name = name,
                type = param.type.resolve().declaration.simpleName.asString(),
                description = paramDocs[name].orEmpty(),
            )
        }
    }

    /**
     * Parse `@param <name> <description>` blocks out of a KDoc comment
     * and return a map from parameter name to description. Descriptions
     * span subsequent indented continuation lines up until the next
     * `@<tag>` or a blank line, matching how Dokka reads KDoc.
     *
     * Returns an empty map when [docString] is null or contains no
     * `@param` directives — callers then fall back to no description,
     * keeping pages for undocumented ops valid.
     */
    private fun parseKDocParams(docString: String?): Map<String, String> {
        if (docString.isNullOrBlank()) return emptyMap()
        val result = linkedMapOf<String, StringBuilder>()
        var current: StringBuilder? = null
        docString.lineSequence().forEach { raw ->
            // KSP hands back the KDoc with leading `*` markers still
            // attached on continuation lines; strip the canonical
            // ` * ` / `*` prefix before pattern-matching.
            val line = raw.trimStart().removePrefix("*").trimStart()
            val paramMatch = Regex("^@param\\s+(\\S+)\\s*(.*)$").matchEntire(line)
            when {
                paramMatch != null -> {
                    val (name, rest) = paramMatch.destructured
                    val sb = StringBuilder(rest.trim())
                    result[name] = sb
                    current = sb
                }
                line.startsWith("@") -> {
                    // Another KDoc tag ends the current @param block.
                    current = null
                }
                line.isBlank() -> {
                    current = null
                }
                else -> {
                    current?.let { sb ->
                        if (sb.isNotEmpty()) sb.append(' ')
                        sb.append(line.trim())
                    }
                }
            }
        }
        return result.mapValues { (_, sb) -> sb.toString().trim() }
            .filterValues { it.isNotEmpty() }
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
            else -> "core"
        }
    }

    /**
     * Canonical SKaiNET version stamped into every generated operator
     * page. Sourced from the `skainet.version` KSP option, which the
     * `skainet-lang-core` build script populates from the root
     * `gradle.properties` `VERSION_NAME` (the same value published to
     * Maven Central). Falls back to `"unknown"` when the option isn't
     * passed — e.g. when the processor is exercised from a unit test
     * fixture that doesn't thread the option through.
     */
    private fun extractVersion(): String =
        options["skainet.version"]?.takeIf { it.isNotBlank() } ?: "unknown"

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
                        append("]")

                        // DARC validation block. Only emitted when an actual
                        // @DarcValidated annotation is present, so unannotated
                        // functions keep the JSON narrow.
                        if (function.validated) {
                            append(",\n")
                            append("          \"validated\": true,\n")
                            append("          \"validatedBy\": \"${escapeJson(function.validatedBy)}\",\n")
                            append("          \"validatedOn\": \"${escapeJson(function.validatedOn)}\",\n")
                            append("          \"validatedCommit\": \"${escapeJson(function.validatedCommit)}\",\n")
                            append("          \"referencesChecked\": ${function.referencesChecked}\n")
                        } else {
                            append("\n")
                        }

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
            logger.error("Failed to generate JSON output: ${e::class.simpleName}: ${e.message}")
        }
    }
}

/**
 * Provider for the OperatorDocProcessor.
 */
class OperatorDocProcessorProvider : SymbolProcessorProvider {
    override fun create(environment: SymbolProcessorEnvironment): SymbolProcessor {
        return OperatorDocProcessor(environment.codeGenerator, environment.logger, environment.options)
    }
}