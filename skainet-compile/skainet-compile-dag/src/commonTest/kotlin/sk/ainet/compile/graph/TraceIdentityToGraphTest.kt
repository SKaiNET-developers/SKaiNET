package sk.ainet.compile.graph

import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.tensorEncoding
import sk.ainet.lang.tensor.ops.tensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1178: what the tape captured survives to the graph. The trace→graph handoff used to drop all
 * metadata — a packed weight arrived downstream as a name string, a parameter as `t7`. Now the
 * `TensorSpec`s carry the encoding *object* and the module-path identity through the same
 * untyped-metadata mechanism `tensorEncoding` proved.
 */
class TraceIdentityToGraphTest {

    private fun ctx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        return DefaultGraphExecutionContext(
            baseOps = DefaultCpuOps(dataFactory),
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) },
        )
    }

    @Test
    fun identityAndEncodingSurviveToTheGraphSpecs() {
        val trainCtx = ctx()
        val id = TensorId(listOf("model", "blk[0]"), "ffn.weight")

        val packed = Q8_0BlockTensorData.fromRawBytes(Shape(32), ByteArray(34))
        @Suppress("UNCHECKED_CAST")
        val weight = trainCtx.fromData(packed as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, FP32::class)
        trainCtx.session.identify(weight, id)

        val dense = trainCtx.fromFloatArray<FP32, Float>(Shape(32), FP32::class, FloatArray(32) { it * 0.5f })

        val tape = DefaultExecutionTape(trainCtx.session)
        tape.startRecording()
        tape.recordOperation(AddOperation<FP32, Float>(), listOf(weight, dense), listOf(dense))
        tape.stopRecording()

        val graph = tape.toComputeGraph(synthesizeExternalInputs = true)
        val allSpecs = graph.nodes.flatMap { it.inputs + it.outputs } + graph.edges.map { it.tensorSpec }

        val identified = allSpecs.filter { it.tensorId == id }
        assertTrue(identified.isNotEmpty(), "the weight's TensorId must reach the graph, got specs: ${allSpecs.map { it.name }}")
        assertEquals(
            TensorEncoding.Q8_0,
            identified.first { it.tensorEncoding != null }.tensorEncoding,
            "the encoding object — block size intact — must ride along",
        )
    }
}
