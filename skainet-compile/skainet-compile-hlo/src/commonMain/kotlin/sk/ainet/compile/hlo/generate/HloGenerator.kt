package sk.ainet.compile.hlo.generate

import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.operators.bind
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
public object HloGenerator {

    /**
     * Generate StableHLO from any [Model] and a sample input tensor.
     *
     * @param model The model whose forward pass will be traced.
     * @param sampleInput A tensor with the desired input shape/dtype (values do not matter).
     * @param functionName The MLIR function name in the emitted module.
     */
    public suspend fun <D : DType, V> generate(
        model: Model<D, V, Tensor<D, V>, Tensor<D, V>>,
        sampleInput: Tensor<D, V>,
        functionName: String = "main",
        /**
         * The compile target (an IREE device name). `null` — the default, byte-identical to the
         * pre-#1180 behaviour — runs no optimization pipeline. A named target runs
         * `dagPipelineFor(target)` with [sk.ainet.compile.opt.passes.LayoutAssignmentPass] as a
         * core pass, plus whatever the target registered via
         * [sk.ainet.compile.opt.TargetOptimizers].
         */
        target: String? = null,
    ): StableHloModule {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val traceInput = sampleInput.bind(ctx)

        // Capture the rebound sample input's tensor ref ID so we can mark it as a function argument.
        @Suppress("UNCHECKED_CAST")
        val inputRefId = ctx.session.refOf(traceInput as Tensor<*, *>).id

        val (tape, _) = ctx.record {
            @Suppress("UNCHECKED_CAST")
            traceForwardPass(
                model as Model<DType, Any?, Tensor<DType, Any?>, Tensor<DType, Any?>>,
                traceInput as Tensor<DType, Any?>,
                ctx
            )
        }

        val computeGraph = tape?.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRefId)
        ) ?: error("Failed to create compute graph: no execution tape was recorded")

        // The optimizer pipeline joins the production path here (#1180): with no target named,
        // nothing runs and the emitted module is what it always was; with one, the layout pass
        // decides and every registered target optimizer gets its say.
        val optimizedGraph = if (target == null) {
            computeGraph
        } else {
            sk.ainet.compile.opt.dagPipelineFor(
                target,
                corePasses = listOf(
                    sk.ainet.compile.opt.passes.LayoutAssignmentPass(target),
                    // SKEEP-005: schedule hints become `skainet.schedule` module metadata.
                    sk.ainet.compile.opt.passes.ScheduleAnnotationPass(target),
                ),
            ).optimize(computeGraph).graph
        }

        val converter = StableHloConverterFactory.createExtended()
        return converter.convert(optimizedGraph, functionName)
    }

    private suspend fun traceForwardPass(
        model: Model<DType, Any?, Tensor<DType, Any?>, Tensor<DType, Any?>>,
        sampleInput: Tensor<DType, Any?>,
        ctx: DefaultGraphExecutionContext
    ) {
        val module = model.create(ctx)
        // Register each parameter's module-path identity on the trace session before recording
        // (#1178): the tensor itself does not know which parameter it is, and refs are immutable
        // once created, so this must happen before the forward pass touches them. Unparsable
        // names stay unidentified rather than guessed.
        for (parameter in module.trainableParameters()) {
            val parsed = runCatching { sk.ainet.lang.tensor.TensorId.parse(parameter.name) }.getOrNull()
            if (parsed != null) ctx.session.identify(parameter.value, parsed)
        }
        model.calculate(
            module = module,
            inputValue = sampleInput,
            executionContext = ctx
        ) { _, _, _ -> }
    }
}
