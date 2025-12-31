package sk.ainet.lang.trace

/**
 * Mock OpAttributeFactory object for testing purposes.
 * This allows generated code to compile during tests.
 */
object OpAttributeFactory {
    
    fun binary(a: Any, b: Any, result: Any): Map<String, Any?> {
        return mapOf(
            "input_shapes" to listOf("tensor", "tensor"),
            "output_shape" to "tensor"
        )
    }
    
    fun unary(input: Any, result: Any): Map<String, Any?> {
        return mapOf(
            "input_shape" to "tensor",
            "output_shape" to "tensor"
        )
    }
    
    fun conv2d(
        input: Any, 
        weight: Any, 
        bias: Any?, 
        result: Any,
        stride: Any,
        padding: Any,
        dilation: Any,
        groups: Any
    ): Map<String, Any?> {
        return mapOf(
            "input_shape" to "tensor",
            "weight_shape" to "tensor",
            "bias_shape" to if (bias != null) "tensor" else null,
            "output_shape" to "tensor",
            "stride" to stride,
            "padding" to padding,
            "dilation" to dilation,
            "groups" to groups
        )
    }
    
    fun shapesAndDTypes(inputs: List<Any>, outputs: List<Any>): Map<String, Any?> {
        return mapOf(
            "input_count" to inputs.size,
            "output_count" to outputs.size
        )
    }
}