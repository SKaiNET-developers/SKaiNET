package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Input tensor operation for graph representation
 */
public class InputOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("input", "input", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.isEmpty()) { "Input operation should not have inputs" }
        throw UnsupportedOperationException("Input operations don't execute - they represent tensor values")
    }

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.isNotEmpty()) {
            return ValidationResult.Invalid(listOf("Input operation should not have inputs, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.isEmpty()) { "Input operation should not have inputs" }
        // This will be set by the caller with the actual tensor spec
        return emptyList()
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = InputOperation<T, V>(newParameters)
}

/**
 * Basic math operations for graph-based execution
 */
public class AddOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("add", "math", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 2) { "Add operation requires exactly 2 inputs" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 2) {
            return ValidationResult.Invalid(listOf("Add operation requires exactly 2 inputs, got ${inputs.size}"))
        }
        val typeA = DType.findByName(inputs[0].dtype)
        val typeB = DType.findByName(inputs[1].dtype)
        if (typeA == null || typeB == null) {
            return ValidationResult.Invalid(listOf("Unknown dtype: ${inputs[0].dtype} or ${inputs[1].dtype}"))
        }
        if (!typeA.isCompatible(typeB)) {
            return ValidationResult.Invalid(listOf("Add operation requires compatible dtypes: ${inputs[0].dtype} and ${inputs[1].dtype}"))
        }
        
        val shapeA = inputs[0].shape
        val shapeB = inputs[1].shape
        if (shapeA != null && shapeB != null) {
            if (!areShapesBroadcastable(shapeA, shapeB)) {
                return ValidationResult.Invalid(listOf("Shapes are not broadcastable: $shapeA and $shapeB"))
            }
        }
        return ValidationResult.Valid
    }

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 2) { "Add operation requires exactly 2 inputs" }
        val typeA = DType.findByName(inputs[0].dtype) ?: throw IllegalArgumentException("Unknown dtype: ${inputs[0].dtype}")
        val typeB = DType.findByName(inputs[1].dtype) ?: throw IllegalArgumentException("Unknown dtype: ${inputs[1].dtype}")
        val resultType = typeA.promoteTo(typeB)
        
        val shapeA = inputs[0].shape
        val shapeB = inputs[1].shape
        val outputShape = if (shapeA != null && shapeB != null) {
            calculateBroadcastShape(shapeA, shapeB)
        } else {
            shapeA ?: shapeB
        }
        
        return listOf(
            TensorSpec(
                name = "add_output",
                shape = outputShape,
                dtype = resultType.name,
                requiresGrad = inputs[0].requiresGrad || inputs[1].requiresGrad
            )
        )
    }

    private fun areShapesBroadcastable(a: List<Int>, b: List<Int>): Boolean {
        val maxLen = maxOf(a.size, b.size)
        for (i in 0 until maxLen) {
            val dimA = if (i < maxLen - a.size) 1 else a[i - (maxLen - a.size)]
            val dimB = if (i < maxLen - b.size) 1 else b[i - (maxLen - b.size)]
            if (dimA != dimB && dimA != 1 && dimB != 1) return false
        }
        return true
    }

    private fun calculateBroadcastShape(a: List<Int>, b: List<Int>): List<Int> {
        val maxLen = maxOf(a.size, b.size)
        return List(maxLen) { i ->
            val dimA = if (i < maxLen - a.size) 1 else a[i - (maxLen - a.size)]
            val dimB = if (i < maxLen - b.size) 1 else b[i - (maxLen - b.size)]
            maxOf(dimA, dimB)
        }
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = AddOperation<T, V>(newParameters)
}

public class SubtractOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("subtract", "math", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 2) { "Subtract operation requires exactly 2 inputs" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 2) {
            return ValidationResult.Invalid(listOf("Subtract operation requires exactly 2 inputs, got ${inputs.size}"))
        }
        val typeA = DType.findByName(inputs[0].dtype)
        val typeB = DType.findByName(inputs[1].dtype)
        if (typeA == null || typeB == null) {
            return ValidationResult.Invalid(listOf("Unknown dtype: ${inputs[0].dtype} or ${inputs[1].dtype}"))
        }
        if (!typeA.isCompatible(typeB)) {
            return ValidationResult.Invalid(listOf("Subtract operation requires compatible dtypes: ${inputs[0].dtype} and ${inputs[1].dtype}"))
        }
        
        val shapeA = inputs[0].shape
        val shapeB = inputs[1].shape
        if (shapeA != null && shapeB != null) {
            if (!areShapesBroadcastable(shapeA, shapeB)) {
                return ValidationResult.Invalid(listOf("Shapes are not broadcastable: $shapeA and $shapeB"))
            }
        }
        return ValidationResult.Valid
    }

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 2) { "Subtract operation requires exactly 2 inputs" }
        val typeA = DType.findByName(inputs[0].dtype) ?: throw IllegalArgumentException("Unknown dtype: ${inputs[0].dtype}")
        val typeB = DType.findByName(inputs[1].dtype) ?: throw IllegalArgumentException("Unknown dtype: ${inputs[1].dtype}")
        val resultType = typeA.promoteTo(typeB)
        
        val shapeA = inputs[0].shape
        val shapeB = inputs[1].shape
        val outputShape = if (shapeA != null && shapeB != null) {
            calculateBroadcastShape(shapeA, shapeB)
        } else {
            shapeA ?: shapeB
        }
        
        return listOf(
            TensorSpec(
                name = "subtract_output",
                shape = outputShape,
                dtype = resultType.name,
                requiresGrad = inputs[0].requiresGrad || inputs[1].requiresGrad
            )
        )
    }

    private fun areShapesBroadcastable(a: List<Int>, b: List<Int>): Boolean {
        val maxLen = maxOf(a.size, b.size)
        for (i in 0 until maxLen) {
            val dimA = if (i < maxLen - a.size) 1 else a[i - (maxLen - a.size)]
            val dimB = if (i < maxLen - b.size) 1 else b[i - (maxLen - b.size)]
            if (dimA != dimB && dimA != 1 && dimB != 1) return false
        }
        return true
    }

    private fun calculateBroadcastShape(a: List<Int>, b: List<Int>): List<Int> {
        val maxLen = maxOf(a.size, b.size)
        return List(maxLen) { i ->
            val dimA = if (i < maxLen - a.size) 1 else a[i - (maxLen - a.size)]
            val dimB = if (i < maxLen - b.size) 1 else b[i - (maxLen - b.size)]
            maxOf(dimA, dimB)
        }
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = SubtractOperation<T, V>(newParameters)
}

public class MultiplyOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("multiply", "math", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 2) { "Multiply operation requires exactly 2 inputs" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 2) {
            return ValidationResult.Invalid(listOf("Multiply operation requires exactly 2 inputs, got ${inputs.size}"))
        }
        val typeA = DType.findByName(inputs[0].dtype)
        val typeB = DType.findByName(inputs[1].dtype)
        if (typeA == null || typeB == null) {
            return ValidationResult.Invalid(listOf("Unknown dtype: ${inputs[0].dtype} or ${inputs[1].dtype}"))
        }
        if (!typeA.isCompatible(typeB)) {
            return ValidationResult.Invalid(listOf("Multiply operation requires compatible dtypes: ${inputs[0].dtype} and ${inputs[1].dtype}"))
        }
        
        val shapeA = inputs[0].shape
        val shapeB = inputs[1].shape
        if (shapeA != null && shapeB != null) {
            if (!areShapesBroadcastable(shapeA, shapeB)) {
                return ValidationResult.Invalid(listOf("Shapes are not broadcastable: $shapeA and $shapeB"))
            }
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 2) { "Multiply operation requires exactly 2 inputs" }
        val typeA = DType.findByName(inputs[0].dtype) ?: throw IllegalArgumentException("Unknown dtype: ${inputs[0].dtype}")
        val typeB = DType.findByName(inputs[1].dtype) ?: throw IllegalArgumentException("Unknown dtype: ${inputs[1].dtype}")
        val resultType = typeA.promoteTo(typeB)
        
        val shapeA = inputs[0].shape
        val shapeB = inputs[1].shape
        val outputShape = if (shapeA != null && shapeB != null) {
            calculateBroadcastShape(shapeA, shapeB)
        } else {
            shapeA ?: shapeB
        }
        
        return listOf(
            TensorSpec(
                name = "multiply_output",
                shape = outputShape,
                dtype = resultType.name,
                requiresGrad = inputs[0].requiresGrad || inputs[1].requiresGrad
            )
        )
    }

    private fun areShapesBroadcastable(a: List<Int>, b: List<Int>): Boolean {
        val maxLen = maxOf(a.size, b.size)
        for (i in 0 until maxLen) {
            val dimA = if (i < maxLen - a.size) 1 else a[i - (maxLen - a.size)]
            val dimB = if (i < maxLen - b.size) 1 else b[i - (maxLen - b.size)]
            if (dimA != dimB && dimA != 1 && dimB != 1) return false
        }
        return true
    }

    private fun calculateBroadcastShape(a: List<Int>, b: List<Int>): List<Int> {
        val maxLen = maxOf(a.size, b.size)
        return List(maxLen) { i ->
            val dimA = if (i < maxLen - a.size) 1 else a[i - (maxLen - a.size)]
            val dimB = if (i < maxLen - b.size) 1 else b[i - (maxLen - b.size)]
            maxOf(dimA, dimB)
        }
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = MultiplyOperation<T, V>(newParameters)
}

public class DivideOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("divide", "math", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 2) { "Divide operation requires exactly 2 inputs" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 2) {
            return ValidationResult.Invalid(listOf("Divide operation requires exactly 2 inputs, got ${inputs.size}"))
        }
        val typeA = DType.findByName(inputs[0].dtype)
        val typeB = DType.findByName(inputs[1].dtype)
        if (typeA == null || typeB == null) {
            return ValidationResult.Invalid(listOf("Unknown dtype: ${inputs[0].dtype} or ${inputs[1].dtype}"))
        }
        if (!typeA.isCompatible(typeB)) {
            return ValidationResult.Invalid(listOf("Divide operation requires compatible dtypes: ${inputs[0].dtype} and ${inputs[1].dtype}"))
        }
        
        val shapeA = inputs[0].shape
        val shapeB = inputs[1].shape
        if (shapeA != null && shapeB != null) {
            if (!areShapesBroadcastable(shapeA, shapeB)) {
                return ValidationResult.Invalid(listOf("Shapes are not broadcastable: $shapeA and $shapeB"))
            }
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 2) { "Divide operation requires exactly 2 inputs" }
        val typeA = DType.findByName(inputs[0].dtype) ?: throw IllegalArgumentException("Unknown dtype: ${inputs[0].dtype}")
        val typeB = DType.findByName(inputs[1].dtype) ?: throw IllegalArgumentException("Unknown dtype: ${inputs[1].dtype}")
        val resultType = typeA.promoteTo(typeB)
        
        val shapeA = inputs[0].shape
        val shapeB = inputs[1].shape
        val outputShape = if (shapeA != null && shapeB != null) {
            calculateBroadcastShape(shapeA, shapeB)
        } else {
            shapeA ?: shapeB
        }
        
        return listOf(
            TensorSpec(
                name = "divide_output",
                shape = outputShape,
                dtype = resultType.name,
                requiresGrad = inputs[0].requiresGrad || inputs[1].requiresGrad
            )
        )
    }

    private fun areShapesBroadcastable(a: List<Int>, b: List<Int>): Boolean {
        val maxLen = maxOf(a.size, b.size)
        for (i in 0 until maxLen) {
            val dimA = if (i < maxLen - a.size) 1 else a[i - (maxLen - a.size)]
            val dimB = if (i < maxLen - b.size) 1 else b[i - (maxLen - b.size)]
            if (dimA != dimB && dimA != 1 && dimB != 1) return false
        }
        return true
    }

    private fun calculateBroadcastShape(a: List<Int>, b: List<Int>): List<Int> {
        val maxLen = maxOf(a.size, b.size)
        return List(maxLen) { i ->
            val dimA = if (i < maxLen - a.size) 1 else a[i - (maxLen - a.size)]
            val dimB = if (i < maxLen - b.size) 1 else b[i - (maxLen - b.size)]
            maxOf(dimA, dimB)
        }
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = DivideOperation<T, V>(newParameters)
}

/**
 * Linear algebra operations
 */
public class MatmulOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("matmul", "linalg", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 2) { "Matmul operation requires exactly 2 inputs" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 2) {
            return ValidationResult.Invalid(listOf("Matmul operation requires exactly 2 inputs, got ${inputs.size}"))
        }
        if (inputs[0].dtype != inputs[1].dtype) {
            return ValidationResult.Invalid(listOf("Matmul operation requires inputs to have same dtype"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 2) { "Matmul operation requires exactly 2 inputs" }
        val outputShape = inputs[0].shape
        return listOf(
            TensorSpec(
                name = "matmul_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad || inputs[1].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = MatmulOperation<T, V>(newParameters)
}

public class TransposeOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("transpose", "linalg", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "Transpose operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("Transpose operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "Transpose operation requires exactly 1 input" }
        val inputShape = inputs[0].shape
        val outputShape = inputShape?.reversed()
        return listOf(
            TensorSpec(
                name = "transpose_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = TransposeOperation<T, V>(newParameters)
}

/**
 * Convolutional operations
 */
public class Conv1dOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("conv1d", "nn", parameters) {

    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size >= 2) { "Conv1d operation requires at least 2 inputs" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size < 2 || inputs.size > 3) {
            return ValidationResult.Invalid(listOf("Conv1d operation requires 2-3 inputs, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size >= 2) { "Conv1d operation requires at least 2 inputs" }
        val inShape = inputs[0].shape
        val wShape = inputs[1].shape
        val stride = (parameters["stride"] as? Int) ?: 1
        val padding = (parameters["padding"] as? Int) ?: 0
        val dilation = (parameters["dilation"] as? Int) ?: 1
        val outputShape = if (inShape != null && wShape != null && inShape.size == 3 && wShape.size == 3) {
            ConvShapeUtils.conv1dOutputShape(
                inShape.toIntArray(), wShape.toIntArray(), stride, padding, dilation
            ).toList()
        } else {
            null
        }
        return listOf(
            TensorSpec(
                name = "conv1d_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs.any { it.requiresGrad }
            )
        )
    }

    override fun clone(newParameters: Map<String, Any>): Operation = Conv1dOperation<T, V>(newParameters)
}

public class Conv2dOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("conv2d", "nn", parameters) {

    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size >= 2) { "Conv2d operation requires at least 2 inputs" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size < 2 || inputs.size > 3) {
            return ValidationResult.Invalid(listOf("Conv2d operation requires 2-3 inputs, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size >= 2) { "Conv2d operation requires at least 2 inputs" }
        val inShape = inputs[0].shape
        val wShape = inputs[1].shape
        val stride = pairParam("stride", 1)
        val padding = pairParam("padding", 0)
        val dilation = pairParam("dilation", 1)
        val outputShape = if (inShape != null && wShape != null && inShape.size == 4 && wShape.size == 4) {
            ConvShapeUtils.conv2dOutputShape(
                inShape.toIntArray(), wShape.toIntArray(), stride, padding, dilation
            ).toList()
        } else {
            null
        }
        return listOf(
            TensorSpec(
                name = "conv2d_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs.any { it.requiresGrad }
            )
        )
    }

    private fun pairParam(name: String, default: Int): Pair<Int, Int> {
        val raw = parameters[name] ?: return default to default
        @Suppress("UNCHECKED_CAST")
        return when (raw) {
            is Pair<*, *> -> raw as Pair<Int, Int>
            is Int -> raw to raw
            else -> default to default
        }
    }

    override fun clone(newParameters: Map<String, Any>): Operation = Conv2dOperation<T, V>(newParameters)
}

public class Conv3dOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("conv3d", "nn", parameters) {

    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size >= 2) { "Conv3d operation requires at least 2 inputs" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size < 2 || inputs.size > 3) {
            return ValidationResult.Invalid(listOf("Conv3d operation requires 2-3 inputs, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size >= 2) { "Conv3d operation requires at least 2 inputs" }
        val inShape = inputs[0].shape
        val wShape = inputs[1].shape
        val stride = tripleParam("stride", 1)
        val padding = tripleParam("padding", 0)
        val dilation = tripleParam("dilation", 1)
        val outputShape = if (inShape != null && wShape != null && inShape.size == 5 && wShape.size == 5) {
            ConvShapeUtils.conv3dOutputShape(
                inShape.toIntArray(), wShape.toIntArray(), stride, padding, dilation
            ).toList()
        } else {
            null
        }
        return listOf(
            TensorSpec(
                name = "conv3d_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs.any { it.requiresGrad }
            )
        )
    }

    private fun tripleParam(name: String, default: Int): Triple<Int, Int, Int> {
        val raw = parameters[name] ?: return Triple(default, default, default)
        @Suppress("UNCHECKED_CAST")
        return when (raw) {
            is Triple<*, *, *> -> raw as Triple<Int, Int, Int>
            is Int -> Triple(raw, raw, raw)
            else -> Triple(default, default, default)
        }
    }

    override fun clone(newParameters: Map<String, Any>): Operation = Conv3dOperation<T, V>(newParameters)
}

/**
 * Pooling operations
 */
public class MaxPool2dOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("maxPool2d", "nn", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "MaxPool2d operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("MaxPool2d operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "MaxPool2d operation requires exactly 1 input" }
        val outputShape = inputs[0].shape
        return listOf(
            TensorSpec(
                name = "maxPool2d_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = MaxPool2dOperation<T, V>(newParameters)
}

public class Upsample2dOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("upsample2d", "nn", parameters) {

    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "Upsample2d operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("Upsample2d operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "Upsample2d operation requires exactly 1 input" }
        val inputShape = inputs[0].shape
        val scale = (parameters["scale"] as? List<*>)?.mapNotNull { (it as? Number)?.toInt() }
        val outputShape = if (inputShape != null && inputShape.size == 4 && scale != null && scale.size == 2) {
            listOf(
                inputShape[0],
                inputShape[1],
                inputShape[2] * scale[0],
                inputShape[3] * scale[1]
            )
        } else {
            inputShape
        }
        return listOf(
            TensorSpec(
                name = "upsample2d_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }

    override fun clone(newParameters: Map<String, Any>): Operation = Upsample2dOperation<T, V>(newParameters)
}

/**
 * Shape operations
 */
public class ReshapeOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("reshape", "shape", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "Reshape operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("Reshape operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "Reshape operation requires exactly 1 input" }
        @Suppress("UNCHECKED_CAST")
        val newShape = parameters["newShape"] as? List<Int>
        return listOf(
            TensorSpec(
                name = "reshape_output",
                shape = newShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = ReshapeOperation<T, V>(newParameters)
}

public class FlattenOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("flatten", "shape", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "Flatten operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("Flatten operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "Flatten operation requires exactly 1 input" }
        val inputShape = inputs[0].shape
        val flattenedSize = inputShape?.fold(1) { acc, dim -> acc * dim }
        val outputShape = if (flattenedSize != null) listOf(flattenedSize) else null
        return listOf(
            TensorSpec(
                name = "flatten_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = FlattenOperation<T, V>(newParameters)
}

/**
 * Activation functions
 */
public class ReluOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("relu", "activation", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "ReLU operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("ReLU operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "ReLU operation requires exactly 1 input" }
        return listOf(
            TensorSpec(
                name = "relu_output",
                shape = inputs[0].shape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = ReluOperation<T, V>(newParameters)
}

public class SoftmaxOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("softmax", "activation", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "Softmax operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("Softmax operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "Softmax operation requires exactly 1 input" }
        return listOf(
            TensorSpec(
                name = "softmax_output",
                shape = inputs[0].shape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = SoftmaxOperation<T, V>(newParameters)
}

public class SigmoidOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("sigmoid", "activation", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "Sigmoid operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("Sigmoid operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "Sigmoid operation requires exactly 1 input" }
        return listOf(
            TensorSpec(
                name = "sigmoid_output",
                shape = inputs[0].shape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = SigmoidOperation<T, V>(newParameters)
}

/**
 * Additional shape operations
 */
public class SqueezeOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("squeeze", "shape", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "Squeeze operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("Squeeze operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "Squeeze operation requires exactly 1 input" }
        val inputShape = inputs[0].shape
        val dimParam = parameters["dim"] as? Int
        val dim = if (dimParam == -1) null else dimParam
        val outputShape = if (inputShape != null) {
            if (dim != null) {
                // Remove specific dimension if it has size 1
                inputShape.toMutableList().apply {
                    if (dim >= 0 && dim < size && this[dim] == 1) {
                        removeAt(dim)
                    }
                }
            } else {
                // Remove all dimensions with size 1
                inputShape.filter { it != 1 }
            }
        } else null
        
        return listOf(
            TensorSpec(
                name = "squeeze_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = SqueezeOperation<T, V>(newParameters)
}

public class UnsqueezeOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("unsqueeze", "shape", parameters) {
    
    override fun <T2 : DType, V2> execute(inputs: List<Tensor<T2, V2>>): List<Tensor<T2, V2>> {
        require(inputs.size == 1) { "Unsqueeze operation requires exactly 1 input" }
        throw UnsupportedOperationException("Direct execution not supported in graph mode")
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        if (inputs.size != 1) {
            return ValidationResult.Invalid(listOf("Unsqueeze operation requires exactly 1 input, got ${inputs.size}"))
        }
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size == 1) { "Unsqueeze operation requires exactly 1 input" }
        val inputShape = inputs[0].shape
        val dim = parameters["dim"] as? Int ?: 0
        val outputShape = if (inputShape != null) {
            inputShape.toMutableList().apply {
                add(if (dim >= 0) dim else size + dim + 1, 1)
            }
        } else null
        
        return listOf(
            TensorSpec(
                name = "unsqueeze_output",
                shape = outputShape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs[0].requiresGrad
            )
        )
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation = UnsqueezeOperation<T, V>(newParameters)
}

/**
 * Scaled dot-product attention operation for tape recording.
 * Output shape = query shape: [batch, nHeads, seqLen, headDim]
 */
public class ScaledDotProductAttentionOperation(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation("scaledDotProductAttention", "nn", parameters) {

    override fun <T : sk.ainet.lang.types.DType, V> execute(
        inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>
    ): List<sk.ainet.lang.tensor.Tensor<T, V>> = emptyList()

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        return if (inputs.size >= 3) ValidationResult.Valid
        else ValidationResult.Invalid(listOf("SDPA requires at least 3 inputs"))
    }

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        require(inputs.size >= 3) { "SDPA requires at least 3 inputs (query, key, value)" }
        return listOf(
            TensorSpec(
                name = "sdpa_output",
                shape = inputs[0].shape,
                dtype = inputs[0].dtype,
                requiresGrad = inputs.any { it.requiresGrad }
            )
        )
    }

    override fun clone(newParameters: Map<String, Any>): Operation =
        ScaledDotProductAttentionOperation(newParameters)
}
