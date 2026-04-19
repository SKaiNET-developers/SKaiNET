package sk.ainet.tape

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.Conv1dOperation
import sk.ainet.lang.tensor.ops.Conv3dOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32

class SimpleExecutionTapeTest {

    private val dataFactory = DenseTensorDataFactory()

    @Test
    fun records_basic_operations_with_wrapper() {
        // Prepare simple FP32 tensors using lang-core facilities
        val shape = Shape(2)
        val a = VoidOpsTensor<FP32, Float>(dataFactory.full(shape, FP32::class, 1.0f), FP32::class)
        val b = VoidOpsTensor<FP32, Float>(dataFactory.full(shape, FP32::class, 2.0f), FP32::class)

        // Use VoidTensorOps as base to avoid backend dependencies
        val baseOps: TensorOps = VoidTensorOps()

        val tape = Execution.withTape {
            val ops = Execution.recordingOps(baseOps)
            val c = ops.add<FP32, Float>(a, b) // should record AddOperation
            // use result so compiler doesn't optimize it away
            val d = ops.relu<FP32, Float>(c)   // should record ReluOperation
        }

        // After scope, recording is stopped
        assertFalse(tape.isRecording, "Tape should not be recording after scope ends")

        // Two operations should be recorded
        assertEquals(2, tape.operations.size, "Expected two recorded operations")

        val first = tape.operations[0]
        assertTrue(first.operation is AddOperation<*, *>, "First op should be AddOperation")
        assertEquals(listOf("a", "b"), first.inputs.map { it.name }, "Stable input names for add")
        assertEquals(listOf("output"), first.outputs.map { it.name }, "Stable output name for unary output")
        // metadata should contain stable-ish tensor ids
        assertTrue(first.inputs.all { it.metadata.containsKey("tid") }, "Inputs should include tensor id metadata")
        assertTrue(first.outputs.all { it.metadata.containsKey("tid") }, "Outputs should include tensor id metadata")

        val second = tape.operations[1]
        assertTrue(second.operation is ReluOperation<*, *>, "Second op should be ReluOperation")
        assertEquals(listOf("input"), second.inputs.map { it.name }, "Stable input name for unary ops")
        assertEquals(listOf("output"), second.outputs.map { it.name })

        // Basic shape and dtype sanity
        assertEquals(shape.dimensions.toList(), first.inputs[0].shape)
        assertEquals("FP32", first.inputs[0].dtype)
    }

    @Test
    fun tape_copy_and_clear_behaviour() {
        val shape = Shape(1)
        val t = VoidOpsTensor<FP32, Float>(dataFactory.zeros(shape, FP32::class), FP32::class)
        val base = VoidTensorOps()

        val tape = Execution.withTape {
            val ops = Execution.recordingOps(base)
            ops.relu<FP32, Float>(t)
        }

        assertEquals(1, tape.operations.size)

        val copy = tape.copy()
        assertEquals(1, copy.operations.size, "Copy should preserve operations")

        tape.clear()
        assertEquals(0, tape.operations.size, "Clear should remove operations")
        // copy remains intact
        assertEquals(1, copy.operations.size)
        assertFalse(copy.isRecording, "Copies keep recording state but not recording by default")
        assertNotNull(copy.operations.first(), "Copy retains recorded op")
    }

    @Test
    fun records_conv1d_through_recording_decorator() {
        val input = VoidOpsTensor<FP32, Float>(
            dataFactory.zeros(Shape(1, 80, 3000), FP32::class), FP32::class
        )
        val weight = VoidOpsTensor<FP32, Float>(
            dataFactory.zeros(Shape(384, 80, 3), FP32::class), FP32::class
        )
        val bias = VoidOpsTensor<FP32, Float>(
            dataFactory.zeros(Shape(384), FP32::class), FP32::class
        )

        val tape = Execution.withTape {
            val ops = Execution.recordingOps(VoidTensorOps())
            ops.conv1d<FP32, Float>(input, weight, bias, stride = 1, padding = 1, dilation = 1, groups = 1)
        }

        val recorded = tape.operations.single { it.operation is Conv1dOperation<*, *> }
        assertEquals(listOf("input", "weight", "bias"), recorded.inputs.map { it.name })
        assertEquals(listOf(1, 80, 3000), recorded.inputs[0].shape)
        assertEquals(listOf(384, 80, 3), recorded.inputs[1].shape)
        assertEquals(listOf(384), recorded.inputs[2].shape)
        assertEquals(listOf(1, 384, 3000), recorded.outputs.single().shape)
        assertEquals(1, recorded.operation.parameters["stride"])
        assertEquals(1, recorded.operation.parameters["padding"])
    }

    @Test
    fun records_conv3d_through_recording_decorator() {
        val input = VoidOpsTensor<FP32, Float>(
            dataFactory.zeros(Shape(1, 3, 16, 16, 16), FP32::class), FP32::class
        )
        val weight = VoidOpsTensor<FP32, Float>(
            dataFactory.zeros(Shape(8, 3, 3, 3, 3), FP32::class), FP32::class
        )

        val tape = Execution.withTape {
            val ops = Execution.recordingOps(VoidTensorOps())
            ops.conv3d<FP32, Float>(
                input, weight, bias = null,
                stride = Triple(1, 1, 1),
                padding = Triple(0, 0, 0),
                dilation = Triple(1, 1, 1),
                groups = 1
            )
        }

        val recorded = tape.operations.single { it.operation is Conv3dOperation<*, *> }
        assertEquals(listOf("input", "weight"), recorded.inputs.map { it.name })
        assertEquals(listOf(1, 3, 16, 16, 16), recorded.inputs[0].shape)
        assertEquals(listOf(8, 3, 3, 3, 3), recorded.inputs[1].shape)
        assertEquals(listOf(1, 8, 14, 14, 14), recorded.outputs.single().shape)
    }
}
