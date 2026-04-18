package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for shape manipulation operations.
 * 
 * This converter implements shape operations using StableHLO primitives:
 * - reshape: using stablehlo.reshape with proper shape inference
 * - flatten: using stablehlo.reshape to flatten specified dimensions
 * - squeeze: using stablehlo.reshape to remove singleton dimensions
 * - unsqueeze: using stablehlo.broadcast_in_dim for dimension expansion
 * 
 * Supports operations as specified in Requirements 2.5:
 * - Shape operations (reshape, flatten, squeeze, unsqueeze)
 * - Dynamic reshaping with runtime shape computation
 * - Proper shape inference and validation
 */
public class ShapeOperationsConverter : StableHloOperationConverter {
    
    override val supportedOperations: Set<String> = setOf(
        "reshape", "flatten", "squeeze", "unsqueeze",
        // Structural tensor ops — generic companions to reshape /
        // flatten / squeeze. concat glues tensors along an axis,
        // slice extracts a static window of a tensor.
        "concat", "concatenate", "cat", "stack",
        "slice"
    )

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            "reshape" -> convertReshape(node, operands, context)
            "flatten" -> convertFlatten(node, operands, context)
            "squeeze" -> convertSqueeze(node, operands, context)
            "unsqueeze" -> convertUnsqueeze(node, operands, context)
            "concat", "concatenate", "cat", "stack" -> convertConcat(node, operands, context)
            "slice" -> convertSlice(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by ShapeOperationsConverter"
            )
        }
    }

    /**
     * Convert concat / concatenate / cat / stack to stablehlo.concatenate.
     *
     * Reads the join axis from `axis` or `dim` parameter (default 0)
     * and emits:
     *
     *     %out = stablehlo.concatenate %a, %b, ..., dim = <axis> : <type>
     */
    private fun convertConcat(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.isEmpty()) {
            return ConversionResult.Failure(
                "Concat operation requires at least 1 operand, got 0",
                "Unsupported concat arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"

        val rank = node.inputs.firstOrNull()?.shape?.size
            ?: outputSpec?.shape?.size ?: 0
        val rawAxis = node.operation.parameters["axis"] as? Int
            ?: node.operation.parameters["dim"] as? Int
            ?: 0
        val axis = if (rawAxis < 0 && rank > 0) rank + rawAxis else rawAxis

        val resultValue = context.nextTempValue()
        val operandList = operands.joinToString(", ")
        val operation = "$resultValue = stablehlo.concatenate $operandList, dim = $axis : $outputType"
        context.emitOperation(operation)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }

    /**
     * Convert slice to stablehlo.slice.
     *
     * Reads per-dim `start_indices`, `limit_indices`, and `strides`
     * from parameters and emits a static slice:
     *
     *     %out = stablehlo.slice %x [s0:l0:d0, s1:l1:d1, ...] : <type>
     *
     * Strides default to 1 per dim when not supplied. Dynamic slice
     * (runtime bounds) is explicitly out of scope for this first pass.
     */
    private fun convertSlice(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Slice operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported slice arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"

        val inputShape = node.inputs.firstOrNull()?.shape ?: emptyList()
        val rank = inputShape.size

        @Suppress("UNCHECKED_CAST")
        val starts = (node.operation.parameters["start_indices"] as? List<Int>)
            ?: (node.operation.parameters["starts"] as? List<Int>)
            ?: List(rank) { 0 }
        @Suppress("UNCHECKED_CAST")
        val limits = (node.operation.parameters["limit_indices"] as? List<Int>)
            ?: (node.operation.parameters["limits"] as? List<Int>)
            ?: inputShape
        @Suppress("UNCHECKED_CAST")
        val strides = (node.operation.parameters["strides"] as? List<Int>)
            ?: List(rank) { 1 }

        val startsAttr = starts.joinToString(", ")
        val limitsAttr = limits.joinToString(", ")
        val stridesAttr = strides.joinToString(", ")

        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.slice ${operands[0]} " +
            "{start_indices = [$startsAttr], " +
            "limit_indices = [$limitsAttr], " +
            "strides = [$stridesAttr]} : $outputType"
        context.emitOperation(operation)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert reshape operation using stablehlo.reshape.
     * Handles both static and dynamic shape specifications.
     */
    private fun convertReshape(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Reshape operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported reshape arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // Get the new shape from parameters or output spec
        val newShape = when {
            outputSpec?.shape != null -> outputSpec.shape
            node.operation.parameters.containsKey("shape") -> {
                @Suppress("UNCHECKED_CAST")
                node.operation.parameters["shape"] as? List<Int>
            }
            node.operation.parameters.containsKey("newShape") -> {
                @Suppress("UNCHECKED_CAST")
                node.operation.parameters["newShape"] as? List<Int>
            }
            else -> null
        }
        
        if (newShape == null || newShape.isEmpty()) {
            return ConversionResult.Failure(
                "Reshape operation requires a target shape specification",
                "Missing shape parameter for reshape node ${node.id}"
            )
        }

        val inputType = resolveOperandType(operands[0], node, context)
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.reshape ${operands[0]} : ($inputType) -> $outputType"
        context.emitOperation(operation)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }

    /**
     * Convert flatten operation using stablehlo.reshape.
     * Flattens dimensions from startDim to endDim into a single dimension.
     */
    private fun convertFlatten(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Flatten operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported flatten arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // Get flatten parameters
        val startDim = node.operation.parameters["startDim"] as? Int ?: 0
        val endDim = node.operation.parameters["endDim"] as? Int ?: -1
        
        context.emitComment("Flatten from dim $startDim to $endDim")

        val inputType = resolveOperandType(operands[0], node, context)
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.reshape ${operands[0]} : ($inputType) -> $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert squeeze operation using stablehlo.reshape.
     * Removes singleton dimensions (dimensions of size 1).
     */
    private fun convertSqueeze(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Squeeze operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported squeeze arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // Get squeeze dimension parameter (null means squeeze all singleton dimensions)
        val dim = node.operation.parameters["dim"] as? Int
        
        if (dim != null) {
            context.emitComment("Squeeze dimension $dim")
        } else {
            context.emitComment("Squeeze all singleton dimensions")
        }

        val inputType = resolveOperandType(operands[0], node, context)
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.reshape ${operands[0]} : ($inputType) -> $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert unsqueeze operation using stablehlo.broadcast_in_dim.
     * Adds a singleton dimension at the specified position.
     */
    private fun convertUnsqueeze(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Unsqueeze operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported unsqueeze arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // Get the dimension to unsqueeze at
        val dim = node.operation.parameters["dim"] as? Int
            ?: return ConversionResult.Failure(
                "Unsqueeze operation requires a 'dim' parameter",
                "Missing dim parameter for unsqueeze node ${node.id}"
            )
        
        context.emitComment("Unsqueeze at dimension $dim")

        // For unsqueeze, we can use either reshape or broadcast_in_dim.
        // Using reshape is simpler for this implementation.
        val inputType = resolveOperandType(operands[0], node, context)
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.reshape ${operands[0]} : ($inputType) -> $outputType"
        context.emitOperation(operation)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }

    /**
     * Look up the MLIR type of an SSA operand.
     *
     * Preference order:
     * 1. Declared type recorded in [ConversionContext.getValueType] —
     *    set either by the function-arg seeder (for `%argN`) or by the
     *    main converter after a prior op succeeded. This is the
     *    operand's actual type at the point of consumption.
     * 2. `node.inputs[0]` — the edge metadata the caller wired.
     * 3. Dynamic fallback — last resort when neither is available.
     *
     * Fixes #518: previous code used `outputType` on both sides of the
     * reshape cast, which produced `(outputShape) -> outputShape` and
     * broke `iree-compile` on any reshape/unsqueeze that consumed a
     * function argument with a different declared shape.
     */
    private fun resolveOperandType(
        operandName: String,
        node: GraphNode,
        context: ConversionContext
    ): String {
        context.getValueType(operandName)?.let { return it }
        node.inputs.firstOrNull()?.let { spec ->
            return context.getTypeMapper().mapTensorType(spec)
        }
        return "tensor<?xf32>"
    }
}