package sk.ainet.compile.minerva

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.tanh
import sk.ainet.compile.export.GraphExportContext

/**
 * Metadata keys understood by the JVM host verifier.
 */
public object MinervaHostVerificationMetadata {
    public const val RUN_CMAKE_BUILD: String = "minerva.hostVerification.runCmakeBuild"
    public const val RUN_CTEST: String = "minerva.hostVerification.runCTest"
    public const val CMAKE_EXECUTABLE: String = "minerva.hostVerification.cmakeExecutable"
    public const val CTEST_EXECUTABLE: String = "minerva.hostVerification.ctestExecutable"
    public const val HOST_OUTPUT_PATH: String = "minerva.hostVerification.hostOutputPath"
    public const val HOST_ADAPTER_SOURCE: String = "minerva.hostVerification.hostAdapterSource"
    public const val HOST_INCLUDE_DIRS: String = "minerva.hostVerification.hostIncludeDirs"
    public const val HOST_LIBRARY_DIRS: String = "minerva.hostVerification.hostLibraryDirs"
    public const val HOST_LIBRARIES: String = "minerva.hostVerification.hostLibraries"
}

/**
 * Status for the overall host verification step and individual substeps.
 */
public enum class MinervaHostVerificationStatus {
    PASSED,
    FAILED,
    SKIPPED
}

/**
 * Input passed to a Minerva host verifier after project packaging.
 */
public data class MinervaHostVerificationRequest(
    public val options: MinervaExportOptions,
    public val intermediate: MinervaIntermediate,
    public val npzModel: MinervaNpzModel,
    public val compilerOutput: MinervaCompilerOutput,
    public val bundle: MinervaExportBundle
) {
    init {
        require(options.projectName == intermediate.projectName) {
            "verification request options and intermediate project names must match"
        }
        require(options.projectName == bundle.projectName) {
            "verification request options and bundle project names must match"
        }
    }
}

/**
 * Host verification result exposed on [MinervaExportResult].
 */
public data class MinervaHostVerification(
    public val status: MinervaHostVerificationStatus,
    public val code: String,
    public val message: String,
    public val hostBuildStatus: MinervaHostVerificationStatus = MinervaHostVerificationStatus.SKIPPED,
    public val hostRunStatus: MinervaHostVerificationStatus = MinervaHostVerificationStatus.SKIPPED,
    public val parityStatus: MinervaHostVerificationStatus = MinervaHostVerificationStatus.SKIPPED,
    public val tolerance: Float = 1.0e-3f,
    public val maxAbsoluteError: Float? = null,
    public val expectedOutput: List<Float> = emptyList(),
    public val observedOutput: List<Float> = emptyList(),
    public val remediation: String = "",
    public val details: Map<String, String> = emptyMap()
) {
    init {
        require(code.isNotBlank()) { "verification code cannot be blank" }
        require(message.isNotBlank()) { "verification message cannot be blank" }
        require(tolerance.isFinite() && tolerance > 0.0f) {
            "verification tolerance must be positive and finite"
        }
        require(maxAbsoluteError == null || maxAbsoluteError.isFinite()) {
            "maxAbsoluteError must be finite when provided"
        }
        require(expectedOutput.all { it.isFinite() }) { "expectedOutput values must be finite" }
        require(observedOutput.all { it.isFinite() }) { "observedOutput values must be finite" }
    }

    public val passed: Boolean
        get() = status == MinervaHostVerificationStatus.PASSED

    public val failed: Boolean
        get() = status == MinervaHostVerificationStatus.FAILED

    public val skipped: Boolean
        get() = status == MinervaHostVerificationStatus.SKIPPED
}

/**
 * Verifies a packaged Minerva project on the host.
 */
public interface MinervaHostVerifier {
    public val backendName: String

    public fun verify(
        request: MinervaHostVerificationRequest,
        context: GraphExportContext
    ): MinervaHostVerification
}

internal object MinervaReferenceEvaluator {
    fun referenceInput(input: MinervaTensorRef): List<Float> {
        val count = input.elementCount
        return List(count) { index -> (index + 1).toFloat() / count.toFloat() }
    }

    fun evaluate(intermediate: MinervaIntermediate, input: List<Float> = referenceInput(intermediate.input)): List<Float> {
        require(input.size == intermediate.input.elementCount) {
            "reference input size must match Minerva input tensor size"
        }
        return intermediate.layers.fold(input) { values, layer ->
            evaluateLayer(values, layer)
        }
    }

    fun maxAbsoluteError(expected: List<Float>, observed: List<Float>): Float {
        require(expected.size == observed.size) { "expected and observed outputs must have the same size" }
        return expected.zip(observed).maxOfOrNull { (left, right) -> abs(left - right) } ?: 0.0f
    }

    private fun evaluateLayer(input: List<Float>, layer: MinervaLayer): List<Float> {
        val weightShape = layer.weights.shape
        require(weightShape.size == 2) {
            "Minerva dense layer weights must be rank-2"
        }
        val inputWidth = weightShape[0]
        val outputWidth = weightShape[1]
        require(input.size % inputWidth == 0) {
            "layer input size must be divisible by the weight input width"
        }
        val weights = requireValues(layer.weights, layer.id)
        val bias = layer.bias?.let { requireValues(it, layer.id) }
        val batchSize = input.size / inputWidth
        val output = MutableList(batchSize * outputWidth) { 0.0f }
        for (batch in 0 until batchSize) {
            for (out in 0 until outputWidth) {
                var sum = bias?.get(out % bias.size) ?: 0.0f
                for (inside in 0 until inputWidth) {
                    sum += input[(batch * inputWidth) + inside] * weights[(inside * outputWidth) + out]
                }
                output[(batch * outputWidth) + out] = activate(sum, layer.activation)
            }
        }
        return output
    }

    private fun requireValues(tensor: MinervaTensorRef, layerId: String): List<Float> {
        return tensor.values ?: error("Tensor '${tensor.id}' on layer '$layerId' does not have numeric values.")
    }

    private fun activate(value: Float, activation: MinervaActivation?): Float {
        return when (activation) {
            null -> value
            MinervaActivation.RELU -> if (value > 0.0f) value else 0.0f
            MinervaActivation.SIGMOID -> (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
            MinervaActivation.TANH -> tanh(value.toDouble()).toFloat()
        }
    }
}
