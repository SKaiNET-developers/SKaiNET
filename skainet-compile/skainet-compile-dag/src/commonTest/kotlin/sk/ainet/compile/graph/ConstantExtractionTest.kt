package sk.ainet.compile.graph

import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.trace.OpTrace
import sk.ainet.lang.trace.PackedConstantException
import sk.ainet.lang.trace.PackedConstantHandling
import sk.ainet.lang.tensor.ops.tensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * #1247 constant-extraction contract:
 * - float weights are ALIASED into the graph, never copied (double residency
 *   of every model weight OOMed the E2B export at a 46 GB heap);
 * - packed frozen params fail loudly by default instead of silently becoming
 *   function arguments (the 190+-arg unservable gemma3n module), with
 *   dequantize-to-FP32 as the opt-in;
 * - narrow-float (BF16/FP16) dense weights widen to one FP32 constant.
 */
class ConstantExtractionTest {

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
    fun float_weight_constant_aliases_the_live_buffer() {
        val trainCtx = ctx()
        val input = trainCtx.fromFloatArray<FP32, Float>(Shape(2, 4), FP32::class, FloatArray(8) { it.toFloat() })
        val weight = trainCtx.fromFloatArray<FP32, Float>(Shape(4, 3), FP32::class, FloatArray(12) { it * 0.5f })
        val output = trainCtx.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class, FloatArray(6))

        val tape = DefaultExecutionTape(trainCtx.session)
        tape.startRecording()
        val inputRef = tape.session.refOf(input)
        val weightRef = tape.session.refOf(weight)
        val outputRef = tape.session.refOf(output)
        tape.recordTrace(
            OpTrace(
                opType = "matmul",
                inputs = listOf(inputRef, weightRef),
                outputs = listOf(outputRef),
                attributes = emptyMap()
            )
        )
        tape.stopRecording()

        val graph = tape.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRef.id)
        )

        val weightNode = graph.nodes.single { it.operation.type == "constant" }
        val embedded = weightNode.operation.parameters["initial_value"] as FloatArray
        val liveBuffer = (weight.data as FloatArrayTensorData).buffer
        assertSame(
            liveBuffer, embedded,
            "the graph constant must alias the live weight buffer, not copy it (#1247 double residency)"
        )
    }

    @Test
    fun packed_frozen_param_fails_loudly_by_default() {
        val trainCtx = ctx()
        val packed = Q8_0BlockTensorData.fromRawBytes(Shape(32), ByteArray(34))
        @Suppress("UNCHECKED_CAST")
        val weight = trainCtx.fromData(packed as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, FP32::class)
        val input = trainCtx.fromFloatArray<FP32, Float>(Shape(32), FP32::class, FloatArray(32))
        val output = trainCtx.fromFloatArray<FP32, Float>(Shape(32), FP32::class, FloatArray(32))

        val tape = DefaultExecutionTape(trainCtx.session)
        tape.startRecording()
        val inputRef = tape.session.refOf(input)
        val weightRef = tape.session.refOf(weight)
        val outputRef = tape.session.refOf(output)
        tape.recordTrace(
            OpTrace(
                opType = "add",
                inputs = listOf(inputRef, weightRef),
                outputs = listOf(outputRef),
                attributes = emptyMap()
            )
        )
        tape.stopRecording()

        val e = assertFailsWith<PackedConstantException> {
            tape.toComputeGraph(
                synthesizeExternalInputs = true,
                inputTensorIds = setOf(inputRef.id)
            )
        }
        val message = e.message ?: ""
        assertTrue("Q8_0" in message, "exception must name the packed encoding: $message")
        assertTrue(weightRef.id in message, "exception must name the tensor: $message")
        assertTrue("DEQUANTIZE" in message, "exception must point at the remediation: $message")
    }

    @Test
    fun packed_frozen_param_dequantizes_when_opted_in() {
        val trainCtx = ctx()
        val packed = Q8_0BlockTensorData.fromRawBytes(Shape(32), ByteArray(34))
        @Suppress("UNCHECKED_CAST")
        val weight = trainCtx.fromData(packed as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, FP32::class)
        val input = trainCtx.fromFloatArray<FP32, Float>(Shape(32), FP32::class, FloatArray(32))
        val output = trainCtx.fromFloatArray<FP32, Float>(Shape(32), FP32::class, FloatArray(32))

        val tape = DefaultExecutionTape(trainCtx.session)
        tape.startRecording()
        val inputRef = tape.session.refOf(input)
        val weightRef = tape.session.refOf(weight)
        val outputRef = tape.session.refOf(output)
        tape.recordTrace(
            OpTrace(
                opType = "add",
                inputs = listOf(inputRef, weightRef),
                outputs = listOf(outputRef),
                attributes = emptyMap()
            )
        )
        tape.stopRecording()

        val graph = tape.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRef.id),
            packedConstants = PackedConstantHandling.DEQUANTIZE
        )

        val weightNode = graph.nodes.single { it.operation.type == "constant" }
        val embedded = weightNode.operation.parameters["initial_value"] as FloatArray
        assertEquals(32, embedded.size, "dequantized constant must carry the full logical extent")
        // The embedded constant is dense FP32 now — the packed encoding must
        // not ride along and misdescribe it.
        val spec = weightNode.outputs.single()
        assertEquals(
            null,
            spec.tensorEncoding,
            "a dequantized constant must not carry the packed encoding"
        )
    }

    @Test
    fun narrow_float_weight_widens_to_fp32_constant() {
        val trainCtx = ctx()
        // 4 bf16 zeros: widening must produce 4 FP32 zeros.
        val bf16 = Bf16DenseTensorData(Shape(4), ByteArray(8))
        @Suppress("UNCHECKED_CAST")
        val weight = trainCtx.fromData(bf16 as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, FP32::class)
        val input = trainCtx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, FloatArray(4))
        val output = trainCtx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, FloatArray(4))

        val tape = DefaultExecutionTape(trainCtx.session)
        tape.startRecording()
        val inputRef = tape.session.refOf(input)
        val weightRef = tape.session.refOf(weight)
        val outputRef = tape.session.refOf(output)
        tape.recordTrace(
            OpTrace(
                opType = "add",
                inputs = listOf(inputRef, weightRef),
                outputs = listOf(outputRef),
                attributes = emptyMap()
            )
        )
        tape.stopRecording()

        val graph = tape.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRef.id)
        )

        val weightNode = graph.nodes.single { it.operation.type == "constant" }
        val embedded = weightNode.operation.parameters["initial_value"] as FloatArray
        assertEquals(4, embedded.size)
        assertTrue(embedded.all { it == 0.0f }, "bf16 zeros must widen to FP32 zeros")
    }

    @Test
    fun forced_input_tensor_still_becomes_input_node() {
        val trainCtx = ctx()
        val input = trainCtx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, FloatArray(2))
        val output = trainCtx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, FloatArray(2))

        val tape = DefaultExecutionTape(trainCtx.session)
        tape.startRecording()
        val inputRef = tape.session.refOf(input)
        val outputRef = tape.session.refOf(output)
        tape.recordTrace(
            OpTrace(
                opType = "relu",
                inputs = listOf(inputRef),
                outputs = listOf(outputRef),
                attributes = emptyMap()
            )
        )
        tape.stopRecording()

        val graph = tape.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRef.id)
        )
        assertTrue(
            graph.nodes.any { it.operation.type == "input" },
            "a tensor in inputTensorIds must remain a function argument even though it is resolvable"
        )
    }
}
