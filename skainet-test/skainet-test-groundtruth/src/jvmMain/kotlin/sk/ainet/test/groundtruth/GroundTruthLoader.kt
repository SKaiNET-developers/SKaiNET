package sk.ainet.test.groundtruth

import kotlinx.io.asSource
import kotlinx.io.buffered
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.GGUFValueType
import sk.ainet.io.gguf.ReaderField
import sk.ainet.io.gguf.ReaderTensor
import sk.ainet.lang.tensor.Shape
import java.io.File
import java.io.InputStream

/**
 * Loads ground truth test cases from GGUF files.
 *
 * GGUF files are expected to contain:
 * - Metadata: description, operation_name (stored as "general.name")
 * - Input tensors: named "input_0", "input_1", etc. or custom names
 * - Result tensor: named "result"
 */
public object GroundTruthLoader {

    /**
     * Loads a single test case from a GGUF file path.
     */
    public fun load(ggufPath: String): GroundTruthTestCase {
        return load(File(ggufPath))
    }

    /**
     * Loads a single test case from a GGUF file.
     */
    public fun load(file: File): GroundTruthTestCase {
        return file.inputStream().use { stream ->
            load(stream, file.absolutePath)
        }
    }

    /**
     * Loads a single test case from an input stream.
     */
    public fun load(inputStream: InputStream, sourcePath: String? = null): GroundTruthTestCase {
        val reader = GGUFReader(
            source = inputStream.asSource().buffered(),
            loadTensorData = true,
            decodeF16ToFloat = true,
            decodeBF16ToFloat = true
        )

        // Extract metadata
        val description = extractStringField(reader, "general.description")
            ?: extractStringField(reader, "description")
            ?: "Unknown test case"

        val operationName = extractStringField(reader, "general.name")
            ?: extractStringField(reader, "name")
            ?: inferOperationFromPath(sourcePath)
            ?: "unknown"

        // Extract tensors
        val inputs = mutableMapOf<String, GroundTruthTensor>()
        var expectedOutput: GroundTruthTensor? = null
        val gradients = mutableMapOf<String, GroundTruthTensor>()

        for (tensor in reader.tensors) {
            val gtTensor = convertToGroundTruthTensor(tensor, reader)

            when {
                tensor.name == "result" || tensor.name == "output" -> {
                    expectedOutput = gtTensor
                }
                tensor.name.startsWith("grad_") -> {
                    val inputName = tensor.name.removePrefix("grad_")
                    gradients[inputName] = gtTensor
                }
                else -> {
                    inputs[tensor.name] = gtTensor
                }
            }
        }

        requireNotNull(expectedOutput) {
            "GGUF file must contain a 'result' or 'output' tensor"
        }

        // Extract test suite and use case from path if available
        val (testSuite, useCase) = extractTestIdentifiers(sourcePath)

        return GroundTruthTestCase(
            description = description,
            operationName = operationName,
            inputs = inputs,
            expectedOutput = expectedOutput,
            expectedGradients = gradients.ifEmpty { null },
            testSuite = testSuite,
            useCase = useCase,
            sourcePath = sourcePath,
            rawOpParams = extractOpParams(reader)
        )
    }

    /**
     * Extract every `op.*` GGUF metadata field into a name -> value map (name with the
     * `op.` prefix stripped). Written by `store_experiment_as_gguf`'s `op_params` handling
     * (`gt/pytorch/io/writer.py`): `add_int32`/`add_uint32` for a scalar int, `add_float32`
     * for a scalar float, `add_array` for a list/tuple of ints (e.g. an asymmetric
     * `(stride_h, stride_w)`) — see [decodeFieldValue] for the corresponding GGUF-side shape.
     */
    private fun extractOpParams(reader: GGUFReader): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        for ((name, field) in reader.fields) {
            if (!name.startsWith("op.")) continue
            decodeFieldValue(field)?.let { result[name.removePrefix("op.")] = it }
        }
        return result
    }

    /**
     * Decode a [sk.ainet.io.gguf.ReaderField] into a plain Int/Float/List<Int>/List<Float>,
     * for the scalar-or-array-of-numbers shape `op.*` fields always use. `field.data` holds
     * indices into `field.parts` for the field's actual value(s) — one index for a scalar,
     * N indices for an N-element array (see GGUFReader.getFieldParts's ARRAY branch); each
     * `field.parts[idx]` is itself a single-element list holding the raw decoded number.
     */
    private fun decodeFieldValue(field: ReaderField): Any? {
        val raw = field.data.mapNotNull { idx -> field.parts.getOrNull(idx)?.firstOrNull() }
        if (raw.isEmpty()) return null
        val numbers = raw.map { toNumber(it) ?: return null }
        val isArray = field.types.firstOrNull() == GGUFValueType.ARRAY
        return if (isArray) numbers else numbers.first()
    }

    private fun toNumber(value: Any): Number? = when (value) {
        is UInt -> value.toInt()
        is ULong -> value.toLong()
        is Number -> value
        else -> null
    }

    /**
     * Loads all test cases from a directory recursively.
     * Finds all .gguf files and loads them as test cases.
     */
    public fun loadFromDirectory(directory: File): List<GroundTruthTestCase> {
        require(directory.isDirectory) { "Path must be a directory: ${directory.absolutePath}" }

        return directory.walkTopDown()
            .filter { it.isFile && it.extension == "gguf" }
            .map { load(it) }
            .toList()
    }

    /**
     * Loads all test cases from a specific test suite directory.
     *
     * @param baseDir Base directory containing test suites (e.g., TS-001, TS-002)
     * @param testSuite Test suite identifier (e.g., "TS-001")
     */
    public fun loadTestSuite(baseDir: File, testSuite: String): List<GroundTruthTestCase> {
        val suiteDir = File(baseDir, testSuite)
        require(suiteDir.isDirectory) { "Test suite directory not found: ${suiteDir.absolutePath}" }
        return loadFromDirectory(suiteDir)
    }

    /**
     * Loads a test case from classpath resources.
     */
    public fun loadFromResource(resourcePath: String): GroundTruthTestCase {
        val stream = GroundTruthLoader::class.java.getResourceAsStream(resourcePath)
            ?: throw IllegalArgumentException("Resource not found: $resourcePath")
        return stream.use { load(it, resourcePath) }
    }

    private fun convertToGroundTruthTensor(tensor: ReaderTensor, reader: GGUFReader): GroundTruthTensor {
        // GGUF stores shape in reversed order (column-major style), convert to row-major
        val shape = Shape(tensor.shape.map { it.toInt() }.reversed().toIntArray())

        // Materialize tensor data if not already loaded
        val rawData = if (tensor.data.isEmpty()) {
            reader.materialize(tensor)
        } else {
            tensor.data
        }

        // Convert to float array
        val floatData = convertToFloatArray(rawData, tensor.tensorType)

        return GroundTruthTensor(
            name = tensor.name,
            shape = shape,
            data = floatData
        )
    }

    @Suppress("UNCHECKED_CAST")
    private fun convertToFloatArray(data: List<Any>, tensorType: GGMLQuantizationType): FloatArray {
        return when (tensorType) {
            GGMLQuantizationType.F32 -> (data as List<Float>).toFloatArray()
            GGMLQuantizationType.F64 -> (data as List<Double>).map { it.toFloat() }.toFloatArray()
            GGMLQuantizationType.F16, GGMLQuantizationType.BF16 -> {
                // Already decoded to Float by GGUFReader
                (data as List<Float>).toFloatArray()
            }
            GGMLQuantizationType.I8 -> (data as List<Byte>).map { it.toFloat() }.toFloatArray()
            GGMLQuantizationType.I16 -> (data as List<Short>).map { it.toFloat() }.toFloatArray()
            GGMLQuantizationType.I32 -> (data as List<Int>).map { it.toFloat() }.toFloatArray()
            GGMLQuantizationType.I64 -> (data as List<Long>).map { it.toFloat() }.toFloatArray()
            else -> throw UnsupportedOperationException(
                "Unsupported tensor type for ground truth: $tensorType"
            )
        }
    }

    private fun extractStringField(reader: GGUFReader, fieldName: String): String? {
        val field = reader.fields[fieldName] ?: return null

        // String fields have structure: [length, data] where data is List<UByte>
        return try {
            if (field.types.contains(GGUFValueType.STRING) && field.parts.size >= 2) {
                val dataIdx = field.data.firstOrNull() ?: return null
                val stringBytes = field.parts.getOrNull(dataIdx) as? List<*> ?: return null
                stringBytes.filterIsInstance<UByte>()
                    .map { it.toByte() }
                    .toByteArray()
                    .decodeToString()
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun inferOperationFromPath(path: String?): String? {
        if (path == null) return null
        val filename = File(path).nameWithoutExtension
        return filename.replace("_", " ").trim()
    }

    private fun extractTestIdentifiers(path: String?): Pair<String?, String?> {
        if (path == null) return null to null

        val parts = path.split(File.separator)
        var testSuite: String? = null
        var useCase: String? = null

        for (part in parts) {
            when {
                part.startsWith("TS-") -> testSuite = part
                part.startsWith("UC-") -> useCase = part.removeSuffix(".gguf")
            }
        }

        return testSuite to useCase
    }
}

/**
 * Extension function to easily get a test case by index.
 */
public operator fun List<GroundTruthTestCase>.get(testSuite: String, useCase: String): GroundTruthTestCase? {
    return find { it.testSuite == testSuite && it.useCase == useCase }
}

/**
 * JVM-specific validator extensions with file system support.
 *
 * `params: OperationParams? = null` (not a bare [OperationParams] default) throughout this
 * file so "not specified" can fall through to each test case's own `op.*`-derived
 * [GroundTruthTestCase.resolvedParams] instead of silently overriding it with an empty one.
 */
public fun GroundTruthValidator.validate(
    ggufPath: String,
    params: OperationParams? = null,
    tolerance: Float? = null,
    rtol: Float = 1e-5f,
    validateGradients: Boolean = false
): GroundTruthValidator.ValidationResult {
    val testCase = GroundTruthLoader.load(ggufPath)
    return validate(testCase, params ?: testCase.resolvedParams(), tolerance, rtol, validateGradients)
}

public fun GroundTruthValidator.validate(
    file: File,
    params: OperationParams? = null,
    tolerance: Float? = null,
    rtol: Float = 1e-5f,
    validateGradients: Boolean = false
): GroundTruthValidator.ValidationResult {
    val testCase = GroundTruthLoader.load(file)
    return validate(testCase, params ?: testCase.resolvedParams(), tolerance, rtol, validateGradients)
}

public fun GroundTruthValidator.validateDirectory(
    directory: File,
    params: OperationParams? = null,
    tolerance: Float? = null,
    rtol: Float = 1e-5f
): List<GroundTruthValidator.ValidationResult> {
    val testCases = GroundTruthLoader.loadFromDirectory(directory)
    return testCases.map { validate(it, params ?: it.resolvedParams(), tolerance, rtol) }
}

public fun GroundTruthValidator.validateTestSuite(
    baseDir: File,
    testSuite: String,
    tolerance: Float? = null,
    rtol: Float = 1e-5f
): List<GroundTruthValidator.ValidationResult> {
    val testCases = GroundTruthLoader.loadTestSuite(baseDir, testSuite)
    return testCases.map { validate(it, tolerance = tolerance, rtol = rtol) }
}

public fun GroundTruthValidator.assertValid(
    ggufPath: String,
    params: OperationParams? = null,
    tolerance: Float? = null,
    rtol: Float = 1e-5f
) {
    val result = validate(ggufPath, params, tolerance, rtol)
    if (!result.success) {
        throw AssertionError(result.toReport())
    }
}
