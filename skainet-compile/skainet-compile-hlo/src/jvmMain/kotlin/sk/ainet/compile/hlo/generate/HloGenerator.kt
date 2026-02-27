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
 * Generates StableHLO MLIR from a registered model by:
 * 1. Creating a tape-recording execution context
 * 2. Running the model forward pass to record operations
 * 3. Converting the execution tape to a ComputeGraph
 * 4. Compiling the graph to StableHLO MLIR
 */
internal object HloGenerator {

    suspend fun generate(descriptor: ModelDescriptor, height: Int, width: Int, batch: Int): StableHloModule {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        val (tape, _) = ctx.record {
            val (model, sampleInput) = descriptor.createModelAndInput(ctx, height, width, batch)
            @Suppress("UNCHECKED_CAST")
            compileModel(
                model as Model<DType, Any?, Tensor<DType, Any?>, Tensor<DType, Any?>>,
                sampleInput as Tensor<DType, Any?>,
                ctx
            )
        }

        val computeGraph = tape?.toComputeGraph()
            ?: error("Failed to create compute graph: no execution tape was recorded")

        return toStableHlo(computeGraph, descriptor.functionName)
    }

    private suspend fun compileModel(
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
