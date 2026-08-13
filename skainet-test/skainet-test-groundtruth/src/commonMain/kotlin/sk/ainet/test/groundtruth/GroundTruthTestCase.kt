package sk.ainet.test.groundtruth

import sk.ainet.lang.tensor.Shape

/**
 * Represents a single ground truth test case loaded from a GGUF file.
 * Contains input tensors, expected output, and metadata about the operation being tested.
 */
public data class GroundTruthTestCase(
    /** Human-readable description of the test case */
    val description: String,

    /** Name of the operation being tested (e.g., "conv2d", "matmul", "relu") */
    val operationName: String,

    /** Input tensors mapped by name */
    val inputs: Map<String, GroundTruthTensor>,

    /** Expected output tensor from PyTorch execution */
    val expectedOutput: GroundTruthTensor,

    /** Optional expected gradients for each input (for backward pass validation) */
    val expectedGradients: Map<String, GroundTruthTensor>? = null,

    /** Test suite identifier (e.g., "TS-001") */
    val testSuite: String? = null,

    /** Use case identifier (e.g., "UC-001") */
    val useCase: String? = null,

    /** Path to the source GGUF file */
    val sourcePath: String? = null,

    /**
     * Operation parameters as written by the Python side's `op.*` GGUF metadata keys
     * (e.g. `op.padding` -> `"padding" to 1`), decoded to Int/Float/List<Int>/List<Float>
     * per value. See [resolvedParams] to turn this into a typed [OperationParams].
     */
    val rawOpParams: Map<String, Any> = emptyMap()
)

/**
 * Maps [GroundTruthTestCase.rawOpParams] onto [OperationParams]'s named, typed fields.
 * A scalar value (`op.padding = 1`) becomes a symmetric pair for the height/width-style
 * params, matching how [OperationParamsBuilder.padding] already turns a single Int into
 * `(value, value)` — the Python side writes symmetric params as a plain scalar, not a
 * 2-element array (see `skainet-ground-truth`'s `op_params={"padding": 1, ...}` usage).
 */
public fun GroundTruthTestCase.resolvedParams(): OperationParams {
    fun pair(name: String): Pair<Int, Int>? = when (val v = rawOpParams[name]) {
        is Int -> v to v
        is Float -> v.toInt() to v.toInt()
        is List<*> -> when (v.size) {
            1 -> (v[0] as Number).toInt().let { it to it }
            else -> (v.getOrNull(0) as? Number)?.toInt()?.let { h ->
                (v.getOrNull(1) as? Number)?.toInt()?.let { w -> h to w }
            }
        }
        else -> null
    }

    fun int(name: String): Int? = (rawOpParams[name] as? Number)?.toInt()
        ?: (rawOpParams[name] as? List<*>)?.firstOrNull().let { (it as? Number)?.toInt() }

    fun float(name: String): Float? = (rawOpParams[name] as? Number)?.toFloat()

    return OperationParams(
        stride = pair("stride"),
        padding = pair("padding"),
        dilation = pair("dilation"),
        groups = int("groups"),
        kernelSize = pair("kernel_size") ?: pair("kernelSize"),
        dim = int("dim"),
        startDim = int("start_dim") ?: int("startDim"),
        endDim = int("end_dim") ?: int("endDim"),
        negativeSlope = float("negative_slope") ?: float("negativeSlope"),
        alpha = float("alpha")
    )
}

/**
 * Represents a tensor loaded from ground truth GGUF file.
 * Stores the raw float data and shape information.
 */
public data class GroundTruthTensor(
    /** Tensor name as stored in GGUF */
    val name: String,

    /** Tensor shape (e.g., [1, 3, 32, 32] for NCHW image) */
    val shape: Shape,

    /** Raw float data in row-major order */
    val data: FloatArray
) {
    /** Number of elements in the tensor */
    val size: Int get() = data.size

    /** Tensor rank (number of dimensions) */
    val rank: Int get() = shape.rank

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GroundTruthTensor) return false
        return name == other.name &&
               shape == other.shape &&
               data.contentEquals(other.data)
    }

    override fun hashCode(): Int {
        var result = name.hashCode()
        result = 31 * result + shape.hashCode()
        result = 31 * result + data.contentHashCode()
        return result
    }

    override fun toString(): String {
        return "GroundTruthTensor(name='$name', shape=${shape.dimensions.contentToString()}, size=$size)"
    }
}

/**
 * Operation parameters extracted from ground truth test case.
 * Used to configure SKaiNET operations to match PyTorch behavior.
 */
public data class OperationParams(
    /** Stride for convolution operations */
    val stride: Pair<Int, Int>? = null,

    /** Padding for convolution/pooling operations */
    val padding: Pair<Int, Int>? = null,

    /** Dilation for convolution operations */
    val dilation: Pair<Int, Int>? = null,

    /** Groups for grouped/depthwise convolution */
    val groups: Int? = null,

    /** Kernel size for pooling operations */
    val kernelSize: Pair<Int, Int>? = null,

    /** Dimension for reduction/softmax operations */
    val dim: Int? = null,

    /** Start dimension for flatten */
    val startDim: Int? = null,

    /** End dimension for flatten */
    val endDim: Int? = null,

    /** Negative slope for LeakyReLU */
    val negativeSlope: Float? = null,

    /** Alpha for ELU */
    val alpha: Float? = null
)
