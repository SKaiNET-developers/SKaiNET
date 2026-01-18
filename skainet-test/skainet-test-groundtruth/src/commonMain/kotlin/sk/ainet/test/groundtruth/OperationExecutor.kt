package sk.ainet.test.groundtruth

import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * Executes SKaiNET operations based on ground truth test cases.
 * Maps operation names to their corresponding TensorOps methods.
 */
public class OperationExecutor(
    private val ops: TensorOps,
    private val dataFactory: TensorDataFactory = DenseTensorDataFactory()
) {

    /**
     * Execute an operation from a ground truth test case.
     *
     * @param testCase The test case containing inputs and operation name
     * @param params Optional operation parameters (stride, padding, etc.)
     * @return The output tensor from SKaiNET
     */
    public fun execute(
        testCase: GroundTruthTestCase,
        params: OperationParams = OperationParams()
    ): Tensor<FP32, Float> {
        // Convert ground truth inputs to SKaiNET tensors
        val inputs = testCase.inputs.mapValues { (_, gtTensor) ->
            createTensor(gtTensor)
        }

        return executeOperation(testCase.operationName, inputs, params)
    }

    /**
     * Execute a named operation with the given inputs.
     */
    public fun executeOperation(
        operationName: String,
        inputs: Map<String, Tensor<FP32, Float>>,
        params: OperationParams = OperationParams()
    ): Tensor<FP32, Float> {
        val normalizedName = normalizeOperationName(operationName)

        return when (normalizedName) {
            // Basic arithmetic
            "add" -> {
                val (a, b) = getTwoInputs(inputs)
                ops.add(a, b)
            }
            "subtract", "sub" -> {
                val (a, b) = getTwoInputs(inputs)
                ops.subtract(a, b)
            }
            "multiply", "mul" -> {
                val (a, b) = getTwoInputs(inputs)
                ops.multiply(a, b)
            }
            "divide", "div" -> {
                val (a, b) = getTwoInputs(inputs)
                ops.divide(a, b)
            }

            // Matrix operations
            "matmul", "mm", "bmm" -> {
                val (a, b) = getTwoInputs(inputs)
                ops.matmul(a, b)
            }
            "transpose" -> {
                val a = getSingleInput(inputs)
                ops.transpose(a)
            }

            // Convolution operations
            "conv2d", "basic_2d_convolution", "strided_convolution", "padded_convolution",
            "dilated_convolution", "depthwise_convolution", "grouped_convolution" -> {
                executeConv2d(inputs, params)
            }
            "conv1d" -> {
                executeConv1d(inputs, params)
            }

            // Pooling operations
            "maxpool2d", "max_pool2d" -> {
                val input = getInput(inputs, "input", "input_0", "x")
                ops.maxPool2d(
                    input,
                    kernelSize = params.kernelSize ?: (2 to 2),
                    stride = params.stride ?: params.kernelSize ?: (2 to 2),
                    padding = params.padding ?: (0 to 0)
                )
            }
            "avgpool2d", "avg_pool2d" -> {
                val input = getInput(inputs, "input", "input_0", "x")
                ops.avgPool2d(
                    input,
                    kernelSize = params.kernelSize ?: (2 to 2),
                    stride = params.stride ?: params.kernelSize ?: (2 to 2),
                    padding = params.padding ?: (0 to 0)
                )
            }

            // Activation functions
            "relu" -> {
                val a = getSingleInput(inputs)
                ops.relu(a)
            }
            "leakyrelu", "leaky_relu" -> {
                val a = getSingleInput(inputs)
                ops.leakyRelu(a, params.negativeSlope ?: 0.01f)
            }
            "elu" -> {
                val a = getSingleInput(inputs)
                ops.elu(a, params.alpha ?: 1.0f)
            }
            "sigmoid" -> {
                val a = getSingleInput(inputs)
                ops.sigmoid(a)
            }
            "silu", "swish" -> {
                val a = getSingleInput(inputs)
                ops.silu(a)
            }
            "gelu" -> {
                val a = getSingleInput(inputs)
                ops.gelu(a)
            }
            "softmax" -> {
                val a = getSingleInput(inputs)
                ops.softmax(a, params.dim ?: -1)
            }
            "logsoftmax", "log_softmax" -> {
                val a = getSingleInput(inputs)
                ops.logSoftmax(a, params.dim ?: -1)
            }

            // Shape operations
            "flatten" -> {
                val a = getSingleInput(inputs)
                ops.flatten(a, params.startDim ?: 0, params.endDim ?: -1)
            }
            "reshape" -> {
                val a = getSingleInput(inputs)
                // Reshape target shape should come from params or expected output
                throw UnsupportedOperationException("Reshape requires target shape parameter")
            }
            "squeeze" -> {
                val a = getSingleInput(inputs)
                ops.squeeze(a, params.dim)
            }
            "unsqueeze" -> {
                val a = getSingleInput(inputs)
                ops.unsqueeze(a, params.dim ?: 0)
            }

            // Reduction operations
            "sum" -> {
                val a = getSingleInput(inputs)
                ops.sum(a, params.dim)
            }
            "mean" -> {
                val a = getSingleInput(inputs)
                ops.mean(a, params.dim)
            }
            "variance", "var" -> {
                val a = getSingleInput(inputs)
                ops.variance(a, params.dim)
            }

            else -> throw UnsupportedOperationException(
                "Operation '$operationName' (normalized: '$normalizedName') is not supported"
            )
        }
    }

    /**
     * Create a SKaiNET tensor from a ground truth tensor.
     */
    public fun createTensor(gtTensor: GroundTruthTensor): Tensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(gtTensor.shape, FP32::class, gtTensor.data)
        return SimpleTensor(data, ops, FP32::class)
    }

    private fun executeConv2d(
        inputs: Map<String, Tensor<FP32, Float>>,
        params: OperationParams
    ): Tensor<FP32, Float> {
        val input = getInput(inputs, "input", "input_0", "x")
        val weight = getInput(inputs, "weight", "input_1", "conv.weight")
        val bias = getInputOrNull(inputs, "bias", "input_2", "conv.bias")

        return ops.conv2d(
            input = input,
            weight = weight,
            bias = bias,
            stride = params.stride ?: (1 to 1),
            padding = params.padding ?: (0 to 0),
            dilation = params.dilation ?: (1 to 1),
            groups = params.groups ?: 1
        )
    }

    private fun executeConv1d(
        inputs: Map<String, Tensor<FP32, Float>>,
        params: OperationParams
    ): Tensor<FP32, Float> {
        val input = getInput(inputs, "input", "input_0", "x")
        val weight = getInput(inputs, "weight", "input_1")
        val bias = getInputOrNull(inputs, "bias", "input_2")

        return ops.conv1d(
            input = input,
            weight = weight,
            bias = bias,
            stride = params.stride?.first ?: 1,
            padding = params.padding?.first ?: 0,
            dilation = params.dilation?.first ?: 1,
            groups = params.groups ?: 1
        )
    }

    private fun normalizeOperationName(name: String): String {
        return name.lowercase()
            .replace(" ", "_")
            .replace("-", "_")
            .trim()
    }

    private fun getSingleInput(inputs: Map<String, Tensor<FP32, Float>>): Tensor<FP32, Float> {
        return when {
            inputs.size == 1 -> inputs.values.first()
            inputs.containsKey("input") -> inputs["input"]!!
            inputs.containsKey("input_0") -> inputs["input_0"]!!
            inputs.containsKey("x") -> inputs["x"]!!
            else -> throw IllegalArgumentException(
                "Cannot determine single input from keys: ${inputs.keys}"
            )
        }
    }

    private fun getTwoInputs(
        inputs: Map<String, Tensor<FP32, Float>>
    ): Pair<Tensor<FP32, Float>, Tensor<FP32, Float>> {
        return when {
            inputs.size == 2 -> {
                val sorted = inputs.entries.sortedBy { it.key }
                sorted[0].value to sorted[1].value
            }
            inputs.containsKey("a") && inputs.containsKey("b") -> {
                inputs["a"]!! to inputs["b"]!!
            }
            inputs.containsKey("input_0") && inputs.containsKey("input_1") -> {
                inputs["input_0"]!! to inputs["input_1"]!!
            }
            else -> throw IllegalArgumentException(
                "Cannot determine two inputs from keys: ${inputs.keys}"
            )
        }
    }

    private fun getInput(
        inputs: Map<String, Tensor<FP32, Float>>,
        vararg names: String
    ): Tensor<FP32, Float> {
        for (name in names) {
            inputs[name]?.let { return it }
        }
        throw IllegalArgumentException(
            "No input found with names: ${names.toList()}, available: ${inputs.keys}"
        )
    }

    private fun getInputOrNull(
        inputs: Map<String, Tensor<FP32, Float>>,
        vararg names: String
    ): Tensor<FP32, Float>? {
        for (name in names) {
            inputs[name]?.let { return it }
        }
        return null
    }

    /**
     * Simple tensor implementation for ground truth validation.
     */
    private class SimpleTensor<T : DType, V>(
        override val data: sk.ainet.lang.tensor.data.TensorData<T, V>,
        private val opsRef: TensorOps,
        override val dtype: KClass<T>
    ) : Tensor<T, V> {
        override val ops: TensorOps get() = opsRef
        override val gradState = GradState<T, V>()
    }
}

/**
 * Builder for creating operation parameters from test case metadata or defaults.
 */
public class OperationParamsBuilder {
    private var stride: Pair<Int, Int>? = null
    private var padding: Pair<Int, Int>? = null
    private var dilation: Pair<Int, Int>? = null
    private var groups: Int? = null
    private var kernelSize: Pair<Int, Int>? = null
    private var dim: Int? = null
    private var startDim: Int? = null
    private var endDim: Int? = null
    private var negativeSlope: Float? = null
    private var alpha: Float? = null

    public fun stride(h: Int, w: Int): OperationParamsBuilder {
        stride = h to w
        return this
    }

    public fun stride(value: Int): OperationParamsBuilder = stride(value, value)

    public fun padding(h: Int, w: Int): OperationParamsBuilder {
        padding = h to w
        return this
    }

    public fun padding(value: Int): OperationParamsBuilder = padding(value, value)

    public fun dilation(h: Int, w: Int): OperationParamsBuilder {
        dilation = h to w
        return this
    }

    public fun dilation(value: Int): OperationParamsBuilder = dilation(value, value)

    public fun groups(value: Int): OperationParamsBuilder {
        groups = value
        return this
    }

    public fun kernelSize(h: Int, w: Int): OperationParamsBuilder {
        kernelSize = h to w
        return this
    }

    public fun kernelSize(value: Int): OperationParamsBuilder = kernelSize(value, value)

    public fun dim(value: Int): OperationParamsBuilder {
        dim = value
        return this
    }

    public fun startDim(value: Int): OperationParamsBuilder {
        startDim = value
        return this
    }

    public fun endDim(value: Int): OperationParamsBuilder {
        endDim = value
        return this
    }

    public fun negativeSlope(value: Float): OperationParamsBuilder {
        negativeSlope = value
        return this
    }

    public fun alpha(value: Float): OperationParamsBuilder {
        alpha = value
        return this
    }

    public fun build(): OperationParams = OperationParams(
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
        kernelSize = kernelSize,
        dim = dim,
        startDim = startDim,
        endDim = endDim,
        negativeSlope = negativeSlope,
        alpha = alpha
    )
}

/**
 * DSL function for building operation parameters.
 */
public fun operationParams(block: OperationParamsBuilder.() -> Unit): OperationParams {
    return OperationParamsBuilder().apply(block).build()
}
