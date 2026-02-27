package sk.ainet.compile.hlo.generate

import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.types.DType

/**
 * Generates StableHLO MLIR from any [Model] by:
 * 1. Creating a tape-recording execution context
 * 2. Running the model forward pass to record operations
 * 3. Converting the execution tape to a ComputeGraph
 * 4. Compiling the graph to StableHLO MLIR
 */
object HloGenerator {

    /**
     * Generate StableHLO from any [Model] and a sample input tensor.
     *
     * @param model       The model whose forward pass will be traced.
     * @param sampleInput A tensor with the desired input shape/dtype (values don't matter).
     * @param functionName The MLIR function name in the emitted module.
     */
    suspend fun <D : DType, V> generate(
        model: Model<D, V, Tensor<D, V>, Tensor<D, V>>,
        sampleInput: Tensor<D, V>,
        functionName: String = "main"
    ): StableHloModule {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        @Suppress("UNCHECKED_CAST")
        val (tape, _) = ctx.record {
            traceForwardPass(
                model as Model<DType, Any?, Tensor<DType, Any?>, Tensor<DType, Any?>>,
                sampleInput as Tensor<DType, Any?>,
                ctx
            )
        }

        val computeGraph = tape?.toComputeGraph()
            ?: error("Failed to create compute graph: no execution tape was recorded")

        return toStableHlo(computeGraph, functionName)
    }

    internal suspend fun generate(descriptor: ModelDescriptor, height: Int, width: Int, batch: Int): StableHloModule {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        val (tape, _) = ctx.record {
            val (model, sampleInput) = descriptor.createModelAndInput(ctx, height, width, batch)
            @Suppress("UNCHECKED_CAST")
            traceForwardPass(
                model as Model<DType, Any?, Tensor<DType, Any?>, Tensor<DType, Any?>>,
                sampleInput as Tensor<DType, Any?>,
                ctx
            )
        }

        val computeGraph = tape?.toComputeGraph()
            ?: error("Failed to create compute graph: no execution tape was recorded")

        return toStableHlo(computeGraph, descriptor.functionName)
    }

    private suspend fun traceForwardPass(
        model: Model<DType, Any?, Tensor<DType, Any?>, Tensor<DType, Any?>>,
        sampleInput: Tensor<DType, Any?>,
        ctx: DefaultGraphExecutionContext
    ) {
        val module = model.create(ctx)
        model.calculate(
            module = module,
            inputValue = sampleInput,
            executionContext = ctx
        ) { _, _, _ -> }
    }
}
