package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for scalar tensor operations: tensor + scalar, tensor * scalar, etc.
 *
 * The KSP-generated tracing wrapper records these as ops named `addScalar` /
 * `subScalar` / `mulScalar` / `divScalar` / `rsubScalar` / `rdivScalar`, with one
 * tensor input and the scalar literal stored in `operation.parameters["b"]`
 * (or `"scalar"`). The StableHLO lowering materializes the scalar as a splat
 * constant of the output type, then applies the corresponding binary op.
 *
 * `rsubScalar(a, x) = a - x` (scalar on the left); `rdivScalar(a, x) = a / x`.
 */
public class ScalarOperationsConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf(
        "addScalar", "subScalar", "mulScalar", "divScalar",
        "rsubScalar", "rdivScalar"
    )

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "${node.operation.name} requires exactly 1 tensor operand, got ${operands.size}",
                "Unsupported ${node.operation.name} arity for node ${node.id}"
            )
        }

        val scalar = extractScalar(node.operation.parameters)
            ?: return ConversionResult.Failure(
                "${node.operation.name} scalar parameter missing on node ${node.id}",
                "Unsupported ${node.operation.name} (missing scalar) for node ${node.id}"
            )

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"

        val (hloOp, reversed) = when (node.operation.name) {
            "addScalar" -> "stablehlo.add" to false
            "subScalar" -> "stablehlo.subtract" to false
            "mulScalar" -> "stablehlo.multiply" to false
            "divScalar" -> "stablehlo.divide" to false
            "rsubScalar" -> "stablehlo.subtract" to true  // scalar - tensor
            "rdivScalar" -> "stablehlo.divide" to true    // scalar / tensor
            else -> return ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by ScalarOperationsConverter"
            )
        }

        val constValue = context.nextTempValue()
        val constOp = "$constValue = stablehlo.constant dense<$scalar> : $outputType"
        context.emitOperation(constOp)

        val resultValue = context.nextTempValue()
        val op = if (reversed) {
            "$resultValue = $hloOp $constValue, ${operands[0]} : $outputType"
        } else {
            "$resultValue = $hloOp ${operands[0]}, $constValue : $outputType"
        }
        context.emitOperation(op)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(constOp, op)
        )
    }

    /**
     * The scalar is stored under "b" by the KSP-generated tracer (see
     * `OpAttributeFactory.scalarOp`). Accept "scalar" as an alias too.
     */
    private fun extractScalar(params: Map<String, Any>): String? {
        val raw = params["b"] ?: params["scalar"] ?: return null
        return when (raw) {
            is Number -> formatFloat(raw.toDouble())
            is String -> raw.toDoubleOrNull()?.let { formatFloat(it) } ?: raw
            else -> raw.toString()
        }
    }

    private fun formatFloat(v: Double): String {
        // Emit as a plain decimal — StableHLO accepts `dense<0.5>`, `dense<1.0e-5>`, etc.
        // Kotlin's default toString uses scientific notation for very small/large numbers,
        // which is also accepted by MLIR.
        return v.toString()
    }
}
