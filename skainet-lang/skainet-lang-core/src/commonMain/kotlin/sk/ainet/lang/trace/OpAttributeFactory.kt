package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Tensor

/**
 * OpAttributeFactory provides methods to generate operation attributes for tracing.
 * It captures metadata about tensor operations for graph construction and optimization.
 */
public object OpAttributeFactory {
    
    /**
     * Generate attributes for binary operations (add, subtract, multiply, etc.).
     */
    public fun binary(a: Tensor<*, *>, b: Tensor<*, *>, result: Tensor<*, *>): Map<String, Any?> {
        return mapOf(
            "input_shapes" to listOf(a.shape.toString(), b.shape.toString()),
            "input_dtypes" to listOf(a.dtype.simpleName, b.dtype.simpleName),
            "output_shape" to result.shape.toString(),
            "output_dtype" to result.dtype.simpleName,
            // Compatibility
            "inputShape" to a.shape.dimensions.toList(),
            "outputShape" to result.shape.dimensions.toList()
        )
    }
    
    /**
     * Generate attributes for unary operations (relu, sigmoid, etc.).
     */
    public fun unary(input: Tensor<*, *>, result: Tensor<*, *>): Map<String, Any?> {
        return mapOf(
            "input_shape" to input.shape.toString(),
            "input_dtype" to input.dtype.simpleName,
            "output_shape" to result.shape.toString(),
            "output_dtype" to result.dtype.simpleName,
            // Compatibility
            "inputShape" to input.shape.dimensions.toList(),
            "outputShape" to result.shape.dimensions.toList()
        )
    }

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
     * Generate attributes for conv2d operations.
     */
    public fun conv2d(
        input: Tensor<*, *>, 
        weight: Tensor<*, *>, 
        bias: Tensor<*, *>?, 
        result: Tensor<*, *>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Map<String, Any?> {
        return mapOf(
            "input_shape" to input.shape.toString(),
            "weight_shape" to weight.shape.toString(),
            "bias_shape" to bias?.shape?.toString(),
            "output_shape" to result.shape.toString(),
            "input_dtype" to input.dtype.simpleName,
            "weight_dtype" to weight.dtype.simpleName,
            "bias_dtype" to bias?.dtype?.simpleName,
            "output_dtype" to result.dtype.simpleName
        )
    }
    
    /**
     * Generate basic shape and dtype attributes for operations.
     */
    public fun shapesAndDTypes(inputs: List<Tensor<*, *>>, outputs: List<Tensor<*, *>>): Map<String, Any?> {
        val attrs = mutableMapOf<String, Any?>(
            "input_shapes" to inputs.map { it.shape.toString() },
            "input_dtypes" to inputs.map { it.dtype.simpleName },
            "output_shapes" to outputs.map { it.shape.toString() },
            "output_dtypes" to outputs.map { it.dtype.simpleName }
        )
        
        // Compatibility for single input/output
        if (inputs.isNotEmpty()) {
            attrs["inputShape"] = inputs.first().shape.dimensions.toList()
        }
        if (outputs.isNotEmpty()) {
            attrs["outputShape"] = outputs.first().shape.dimensions.toList()
        }
        
        return attrs
    }
}