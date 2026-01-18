package sk.ainet.test.groundtruth

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import kotlin.math.abs
import kotlin.math.max

/**
 * Tolerance configuration for tensor comparisons.
 * Different operations may require different tolerances due to numerical precision.
 */
public object ToleranceConfig {
    /** Very strict tolerance for exact arithmetic operations */
    public const val STRICT: Float = 1e-6f

    /** Standard tolerance for most operations */
    public const val STANDARD: Float = 1e-5f

    /** Relaxed tolerance for transcendental functions (exp, log, etc.) */
    public const val RELAXED: Float = 1e-4f

    /** Tolerance for gradient computations (accumulated errors) */
    public const val GRADIENT: Float = 1e-4f

    /** Very relaxed tolerance for complex operations with many floating point ops */
    public const val VERY_RELAXED: Float = 1e-3f

    /**
     * Get recommended tolerance for a given operation.
     */
    public fun forOperation(operationName: String): Float {
        return when {
            operationName in listOf("add", "subtract", "multiply", "divide") -> STRICT
            operationName in listOf("matmul", "conv2d", "conv1d", "conv3d") -> STANDARD
            operationName in listOf("relu", "leakyRelu", "flatten", "reshape") -> STRICT
            operationName in listOf("sigmoid", "gelu", "silu", "softmax", "logSoftmax") -> RELAXED
            operationName in listOf("sum", "mean", "variance") -> STANDARD
            operationName.contains("grad") -> GRADIENT
            else -> STANDARD
        }
    }
}

/**
 * Result of a tensor comparison, containing detailed information about any mismatches.
 */
public data class ComparisonResult(
    /** Whether the tensors are considered equal within tolerance */
    val success: Boolean,

    /** Maximum absolute difference found */
    val maxAbsDiff: Float,

    /** Maximum relative difference found */
    val maxRelDiff: Float,

    /** Number of elements that exceeded tolerance */
    val mismatchCount: Int,

    /** Total number of elements compared */
    val totalElements: Int,

    /** Index of first mismatched element (if any) */
    val firstMismatchIndex: Int?,

    /** Expected value at first mismatch */
    val firstMismatchExpected: Float?,

    /** Actual value at first mismatch */
    val firstMismatchActual: Float?,

    /** Shape comparison result */
    val shapesMatch: Boolean,

    /** Expected shape */
    val expectedShape: Shape,

    /** Actual shape */
    val actualShape: Shape
) {
    /**
     * Percentage of elements that matched within tolerance.
     */
    val matchPercentage: Float
        get() = if (totalElements > 0) {
            (totalElements - mismatchCount).toFloat() / totalElements * 100f
        } else {
            100f
        }

    /**
     * Create a detailed error message for failed comparisons.
     */
    public fun toErrorMessage(): String {
        if (success) return "Comparison successful"

        val sb = StringBuilder()
        sb.appendLine("Tensor comparison failed:")

        if (!shapesMatch) {
            sb.appendLine("  Shape mismatch: expected ${expectedShape.dimensions.contentToString()}, got ${actualShape.dimensions.contentToString()}")
        }

        sb.appendLine("  Max absolute difference: $maxAbsDiff")
        sb.appendLine("  Max relative difference: $maxRelDiff")
        sb.appendLine("  Mismatched elements: $mismatchCount / $totalElements (${formatPercent(100f - matchPercentage)}%)")

        if (firstMismatchIndex != null) {
            sb.appendLine("  First mismatch at index $firstMismatchIndex: expected $firstMismatchExpected, got $firstMismatchActual")
        }

        return sb.toString()
    }

    private fun formatPercent(value: Float): String {
        return ((value * 100).toInt() / 100.0).toString()
    }
}

/**
 * Utility object for comparing tensors with configurable tolerances.
 */
public object TensorAssertions {

    /**
     * Compare two float arrays with absolute and relative tolerance.
     * Uses the formula: |a - b| <= atol + rtol * max(|a|, |b|)
     */
    public fun compare(
        expected: FloatArray,
        actual: FloatArray,
        expectedShape: Shape,
        actualShape: Shape,
        atol: Float = ToleranceConfig.STANDARD,
        rtol: Float = 1e-5f
    ): ComparisonResult {
        val shapesMatch = expectedShape.dimensions.contentEquals(actualShape.dimensions)

        if (!shapesMatch) {
            return ComparisonResult(
                success = false,
                maxAbsDiff = Float.MAX_VALUE,
                maxRelDiff = Float.MAX_VALUE,
                mismatchCount = max(expected.size, actual.size),
                totalElements = max(expected.size, actual.size),
                firstMismatchIndex = null,
                firstMismatchExpected = null,
                firstMismatchActual = null,
                shapesMatch = false,
                expectedShape = expectedShape,
                actualShape = actualShape
            )
        }

        if (expected.size != actual.size) {
            return ComparisonResult(
                success = false,
                maxAbsDiff = Float.MAX_VALUE,
                maxRelDiff = Float.MAX_VALUE,
                mismatchCount = max(expected.size, actual.size),
                totalElements = max(expected.size, actual.size),
                firstMismatchIndex = null,
                firstMismatchExpected = null,
                firstMismatchActual = null,
                shapesMatch = true,
                expectedShape = expectedShape,
                actualShape = actualShape
            )
        }

        var maxAbsDiff = 0f
        var maxRelDiff = 0f
        var mismatchCount = 0
        var firstMismatchIdx: Int? = null
        var firstMismatchExp: Float? = null
        var firstMismatchAct: Float? = null

        for (i in expected.indices) {
            val exp = expected[i]
            val act = actual[i]
            val absDiff = abs(exp - act)

            // Handle special cases
            val isClose = when {
                exp.isNaN() && act.isNaN() -> true
                exp.isInfinite() && act.isInfinite() && exp == act -> true
                exp.isNaN() || act.isNaN() -> false
                exp.isInfinite() || act.isInfinite() -> false
                else -> {
                    val maxAbs = max(abs(exp), abs(act))
                    val tolerance = atol + rtol * maxAbs
                    absDiff <= tolerance
                }
            }

            maxAbsDiff = max(maxAbsDiff, absDiff)

            val relDiff = if (abs(exp) > 1e-10f) absDiff / abs(exp) else absDiff
            maxRelDiff = max(maxRelDiff, relDiff)

            if (!isClose) {
                mismatchCount++
                if (firstMismatchIdx == null) {
                    firstMismatchIdx = i
                    firstMismatchExp = exp
                    firstMismatchAct = act
                }
            }
        }

        return ComparisonResult(
            success = mismatchCount == 0,
            maxAbsDiff = maxAbsDiff,
            maxRelDiff = maxRelDiff,
            mismatchCount = mismatchCount,
            totalElements = expected.size,
            firstMismatchIndex = firstMismatchIdx,
            firstMismatchExpected = firstMismatchExp,
            firstMismatchActual = firstMismatchAct,
            shapesMatch = true,
            expectedShape = expectedShape,
            actualShape = actualShape
        )
    }

    /**
     * Compare a SKaiNET tensor against a ground truth tensor.
     */
    public fun <T : DType, V> compare(
        expected: GroundTruthTensor,
        actual: Tensor<T, V>,
        atol: Float = ToleranceConfig.STANDARD,
        rtol: Float = 1e-5f
    ): ComparisonResult {
        val actualData = extractFloatData(actual)
        return compare(
            expected = expected.data,
            actual = actualData,
            expectedShape = expected.shape,
            actualShape = actual.shape,
            atol = atol,
            rtol = rtol
        )
    }

    /**
     * Compare two GroundTruthTensors.
     */
    public fun compare(
        expected: GroundTruthTensor,
        actual: GroundTruthTensor,
        atol: Float = ToleranceConfig.STANDARD,
        rtol: Float = 1e-5f
    ): ComparisonResult {
        return compare(
            expected = expected.data,
            actual = actual.data,
            expectedShape = expected.shape,
            actualShape = actual.shape,
            atol = atol,
            rtol = rtol
        )
    }

    /**
     * Assert that a tensor matches the expected ground truth.
     * Throws AssertionError with detailed message if comparison fails.
     */
    public fun <T : DType, V> assertTensorClose(
        expected: GroundTruthTensor,
        actual: Tensor<T, V>,
        atol: Float = ToleranceConfig.STANDARD,
        rtol: Float = 1e-5f,
        message: String? = null
    ) {
        val result = compare(expected, actual, atol, rtol)
        if (!result.success) {
            val prefix = if (message != null) "$message\n" else ""
            throw AssertionError(prefix + result.toErrorMessage())
        }
    }

    /**
     * Assert that two float arrays are close within tolerance.
     */
    public fun assertArrayClose(
        expected: FloatArray,
        actual: FloatArray,
        expectedShape: Shape,
        actualShape: Shape,
        atol: Float = ToleranceConfig.STANDARD,
        rtol: Float = 1e-5f,
        message: String? = null
    ) {
        val result = compare(expected, actual, expectedShape, actualShape, atol, rtol)
        if (!result.success) {
            val prefix = if (message != null) "$message\n" else ""
            throw AssertionError(prefix + result.toErrorMessage())
        }
    }

    /**
     * Assert that shapes match exactly.
     */
    public fun assertShapeEquals(expected: Shape, actual: Shape, message: String? = null) {
        if (!expected.dimensions.contentEquals(actual.dimensions)) {
            val prefix = if (message != null) "$message: " else ""
            throw AssertionError(
                "${prefix}Shape mismatch: expected ${expected.dimensions.contentToString()}, " +
                "got ${actual.dimensions.contentToString()}"
            )
        }
    }

    /**
     * Extract float data from a SKaiNET tensor.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> extractFloatData(tensor: Tensor<T, V>): FloatArray {
        val data = tensor.data

        // Try direct buffer access for FP32 tensors
        if (data is FloatArrayTensorData<*>) {
            return data.buffer.copyOf()
        }

        // Fallback: iterate through all elements
        val shape = tensor.shape
        val result = FloatArray(shape.volume)
        val indices = IntArray(shape.rank)

        for (i in result.indices) {
            // Convert linear index to multi-dimensional indices
            var remaining = i
            for (d in shape.rank - 1 downTo 0) {
                indices[d] = remaining % shape.dimensions[d]
                remaining /= shape.dimensions[d]
            }

            val value = data.get(*indices)
            result[i] = when (value) {
                is Float -> value
                is Double -> value.toFloat()
                is Int -> value.toFloat()
                is Long -> value.toFloat()
                is Short -> value.toFloat()
                is Byte -> value.toFloat()
                else -> throw UnsupportedOperationException(
                    "Cannot convert ${value?.let { it::class }} to Float"
                )
            }
        }

        return result
    }
}

/**
 * Extension function for convenient tensor comparison in tests.
 */
public infix fun <T : DType, V> Tensor<T, V>.shouldBeCloseTo(expected: GroundTruthTensor) {
    TensorAssertions.assertTensorClose(expected, this)
}

/**
 * Extension function with custom tolerance.
 */
public fun <T : DType, V> Tensor<T, V>.shouldBeCloseTo(
    expected: GroundTruthTensor,
    atol: Float,
    rtol: Float = 1e-5f
) {
    TensorAssertions.assertTensorClose(expected, this, atol, rtol)
}
