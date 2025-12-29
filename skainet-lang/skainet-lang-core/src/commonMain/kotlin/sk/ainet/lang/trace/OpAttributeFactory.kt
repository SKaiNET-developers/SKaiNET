package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Tensor

/**
 * OpAttributeFactory provides methods to generate operation attributes for tracing.
 * It captures metadata about tensor operations for graph construction and optimization.
 */
public object OpAttributeFactory {
    
    /** Binary op convenience (e.g., add, mul). */
    public fun binary(
        a: Tensor<*, *>, b: Tensor<*, *>, result: Tensor<*, *>
    ): Map<String, Any?> = shapesAndDTypes(listOf(a, b), listOf(result))

    /** Unary op convenience (e.g., relu, sigmoid). */
    public fun unary(
        input: Tensor<*, *>, result: Tensor<*, *>
    ): Map<String, Any?> = shapesAndDTypes(listOf(input), listOf(result))

    /** Conv2d op attributes: shapes/dtypes + stride/padding/dilation/groups and bias flag. */
    public fun conv2d(
        input: Tensor<*, *>,
        weight: Tensor<*, *>,
        bias: Tensor<*, *>?,
        result: Tensor<*, *>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Map<String, Any?> = shapesAndDTypes(listOf(input, weight) + listOfNotNull(bias), listOf(result)) + mapOf(
        "stride" to listOf(stride.first, stride.second),
        "padding" to listOf(padding.first, padding.second),
        "dilation" to listOf(dilation.first, dilation.second),
        "groups" to groups,
        "hasBias" to (bias != null)
    )

    /**
     * Generate attributes for scalar operations.
     */
    public fun scalarOp(
        input: Tensor<*, *>,
        scalar: Number,
        result: Tensor<*, *>,
        isReversed: Boolean
    ): Map<String, Any?> {
        return mapOf(
            "scalar" to scalar,
            "input_shape" to input.shape.toString(),
            "input_dtype" to input.dtype.simpleName,
            "output_shape" to result.shape.toString(),
            "output_dtype" to result.dtype.simpleName,
            "is_reversed" to isReversed,
            // Compatibility
            "inputShape" to input.shape.dimensions.toList(),
            "outputShape" to result.shape.dimensions.toList()
        )
    }

    /**
     * Generate basic shape and dtype attributes for operations.
     */
    public fun shapesAndDTypes(inputs: List<Tensor<*, *>>, outputs: List<Tensor<*, *>>): Map<String, Any?> {
        val attrs = mutableMapOf<String, Any?>(
            "input_shapes" to inputs.map { it.shape.toString() },
            "input_dtypes" to inputs.map { it.dtype.simpleName ?: it.dtype.toString() },
            "output_shapes" to outputs.map { it.shape.toString() },
            "output_dtypes" to outputs.map { it.dtype.simpleName ?: it.dtype.toString() },
            // Compatibility for skainet-compile-core
            "inputShapes" to inputs.map { it.shape.dimensions.toList() },
            "outputShapes" to outputs.map { it.shape.dimensions.toList() },
            "inputDTypes" to inputs.map { it.dtype.simpleName ?: it.dtype.toString() },
            "outputDTypes" to outputs.map { it.dtype.simpleName ?: it.dtype.toString() }
        )
        
        // Compatibility for single input/output
        if (inputs.isNotEmpty()) {
            attrs["inputShape"] = inputs.first().shape.dimensions.toList()
        } else {
            attrs["inputShape"] = emptyList<Int>()
        }
        if (outputs.isNotEmpty()) {
            attrs["outputShape"] = outputs.first().shape.dimensions.toList()
        } else {
            attrs["outputShape"] = emptyList<Int>()
        }
        
        return attrs
    }
}