package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for elementwise unary math operations that map 1:1 onto a StableHLO primitive.
 *
 * Without these, traced graphs that use sqrt/exp/abs/… fall through to the "no converter found"
 * path in [sk.ainet.compile.hlo.StableHloConverter]. Because that path never calls
 * `context.setValueName`, downstream consumers of the failed node lose an operand via the
 * `mapNotNull` in `processNode`, and appear to the next converter as if they had "wrong arity".
 * A single missing unary converter therefore cascades many unrelated "Unsupported X arity"
 * errors further down the graph — which is most of what the Whisper upstream-issues doc calls
 * "Issue B".
 */
public class UnaryMathConverter : StableHloOperationConverter {

    // Trace op-name → StableHLO op name
    private val opMap: Map<String, String> = mapOf(
        "sqrt" to "stablehlo.sqrt",
        "rsqrt" to "stablehlo.rsqrt",
        "exp" to "stablehlo.exponential",
        "expm1" to "stablehlo.exponential_minus_one",
        "log" to "stablehlo.log",
        "log1p" to "stablehlo.log_plus_one",
        "abs" to "stablehlo.abs",
        "sign" to "stablehlo.sign",
        "negate" to "stablehlo.negate",
        "neg" to "stablehlo.negate",
        "ceil" to "stablehlo.ceil",
        "floor" to "stablehlo.floor",
        "round" to "stablehlo.round_nearest_even",
        "cos" to "stablehlo.cosine",
        "sin" to "stablehlo.sine",
    )

    override val supportedOperations: Set<String> = opMap.keys

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        val name = node.operation.name.lowercase()
        val hloOp = opMap[name] ?: return ConversionResult.Unsupported(
            node.operation.name,
            "Operation not supported by UnaryMathConverter"
        )

        if (operands.size != 1) {
            return ConversionResult.Failure(
                "$name operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported $name arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"

        val resultValue = context.nextTempValue()
        val op = "$resultValue = $hloOp ${operands[0]} : $outputType"
        context.emitOperation(op)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(op)
        )
    }
}
