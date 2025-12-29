package sk.ainet.exec

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.ops.KspTensorOps
import sk.ainet.lang.types.FP32

/**
 * Integration tests for TracingTensorOps with DefaultGraphExecutionContext.
 * 
 * These tests verify that the generated TracingTensorOps wrapper works correctly
 * with real TensorOps implementations and that OpTrace emission functions properly
 * with the existing tracing infrastructure.
 */
class KspTracingIntegrationTest {

    private val dataFactory = DenseTensorDataFactory()

    @Suppress("UNCHECKED_CAST")
    private fun ones(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.ones<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    @Test
    fun kspTracingTensorOps_isUsedByDefaultGraphExecutionContext() {
        // Test that DefaultGraphExecutionContext now uses TracingTensorOps instead of manual TracingTensorOps
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        ctx.startRecording()
        
        val ops = ctx.ops
        
        // Verify that the ops instance is KspTensorOps
        assertTrue(ops is KspTensorOps, "DefaultGraphExecutionContext should use KspTensorOps")
        
        ctx.stopRecording()
    }

    @Test
    fun kspTracingTensorOps_delegatesCorrectlyToBaseImplementation() {
        // Test that TracingTensorOps properly delegates to base implementation
        val ctx = DefaultGraphExecutionContext.eager(DefaultCpuOps(dataFactory))
        val ops = ctx.ops
        
        val a = ones(Shape(intArrayOf(2, 2)))
        val b = ones(Shape(intArrayOf(2, 2)))
        
        // Test basic arithmetic operations
        val sum = ops.add(a, b)
        val product = ops.multiply(a, b)
        val difference = ops.subtract(a, b)
        
        // Verify numeric results are correct (delegation works)
        assertEquals(2.0f, sum.data.get(0, 0), 1e-6f, "Addition should work correctly")
        assertEquals(1.0f, product.data.get(0, 0), 1e-6f, "Multiplication should work correctly")
        assertEquals(0.0f, difference.data.get(0, 0), 1e-6f, "Subtraction should work correctly")
    }

    @Test
    fun kspTracingTensorOps_emitsOpTracesCorrectly() {
        // Test that TracingTensorOps emits OpTrace events correctly
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        ctx.startRecording()
        val ops = ctx.ops
        
        val a = ones(Shape(intArrayOf(1)))
        val b = ones(Shape(intArrayOf(1)))
        
        // Execute operations that should be traced
        val sum = ops.add(a, b)
        val relu_result = ops.relu(sum)
        val sigmoid_result = ops.sigmoid(relu_result)
        
        val tape = ctx.stopRecording()
        assertNotNull(tape)
        tape as DefaultExecutionTape
        
        val traces = tape.traces
        assertTrue(traces.isNotEmpty(), "Tape should contain traces")
        
        // Verify specific operations are traced
        val opTypes = traces.map { it.opType }.toSet()
        assertTrue("add" in opTypes, "Should trace add operation")
        assertTrue("relu" in opTypes, "Should trace relu operation")
        assertTrue("sigmoid" in opTypes, "Should trace sigmoid operation")
        
        // Verify trace structure
        val addTrace = traces.first { it.opType == "add" }
        assertEquals(2, addTrace.inputs.size, "Add should have 2 inputs")
        assertEquals(1, addTrace.outputs.size, "Add should have 1 output")
        assertTrue(addTrace.attributes.isNotEmpty(), "Add should have attributes")
    }

    @Test
    fun kspTracingTensorOps_handlesMultiOutputOperations() {
        // Test that TracingTensorOps correctly handles operations that return multiple tensors
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        ctx.startRecording()
        val ops = ctx.ops
        
        val input = ones(Shape(intArrayOf(4)))
        
        // Test split operation (returns List<Tensor>)
        val splits = ops.split(input, 2, 0)
        
        val tape = ctx.stopRecording()
        assertNotNull(tape)
        tape as DefaultExecutionTape
        
        val traces = tape.traces
        val splitTrace = traces.firstOrNull { it.opType == "split" }
        assertNotNull(splitTrace, "Should have split trace")
        
        assertEquals(1, splitTrace.inputs.size, "Split should have 1 input")
        assertEquals(2, splitTrace.outputs.size, "Split should have 2 outputs")
        assertTrue(splitTrace.attributes.containsKey("splitSize"), "Split should have splitSize attribute")
        assertTrue(splitTrace.attributes.containsKey("dim"), "Split should have dim attribute")
    }

    @Test
    fun kspTracingTensorOps_handlesScalarOperations() {
        // Test that TracingTensorOps correctly handles scalar operations
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        ctx.startRecording()
        val ops = ctx.ops
        
        val tensor = ones(Shape(intArrayOf(2, 2)))
        
        // Test scalar operations
        val addScalar = ops.addScalar(tensor, 5.0)
        val mulScalar = ops.mulScalar(tensor, 2.0)
        
        val tape = ctx.stopRecording()
        assertNotNull(tape)
        tape as DefaultExecutionTape
        
        val traces = tape.traces
        val scalarOps = traces.filter { it.opType in setOf("addScalar", "mulScalar") }
        assertEquals(2, scalarOps.size, "Should have 2 scalar operation traces")
        
        // Verify scalar operations have correct attributes
        val addScalarTrace = traces.first { it.opType == "addScalar" }
        assertTrue(addScalarTrace.attributes.containsKey("scalar"), "addScalar should have scalar attribute")
        assertEquals(5.0, addScalarTrace.attributes["scalar"], "addScalar should have correct scalar value")
    }

    @Test
    fun kspTracingTensorOps_handlesShapeOperations() {
        // Test that TracingTensorOps correctly handles shape manipulation operations
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        ctx.startRecording()
        val ops = ctx.ops
        
        val tensor = ones(Shape(intArrayOf(2, 2)))
        
        // Test shape operations
        val reshaped = ops.reshape(tensor, Shape(intArrayOf(4)))
        val flattened = ops.flatten(tensor, 0, 1)
        val unsqueezed = ops.unsqueeze(tensor, 0)
        
        val tape = ctx.stopRecording()
        assertNotNull(tape)
        tape as DefaultExecutionTape
        
        val traces = tape.traces
        val shapeOps = traces.filter { it.opType in setOf("reshape", "flatten", "unsqueeze") }
        assertEquals(3, shapeOps.size, "Should have 3 shape operation traces")
        
        // Verify shape operations have correct attributes
        val reshapeTrace = traces.first { it.opType == "reshape" }
        assertTrue(reshapeTrace.attributes.containsKey("inputShape"), "reshape should have inputShape")
        assertTrue(reshapeTrace.attributes.containsKey("outputShape"), "reshape should have outputShape")
    }

    @Test
    fun kspTracingTensorOps_compatibleWithExistingTracingInfrastructure() {
        // Test that TracingTensorOps works with existing tracing infrastructure (graphs, composite sinks)
        val graph = DefaultComputeGraph()
        val ctx = DefaultGraphExecutionContext.tapeAndGraph(DefaultCpuOps(dataFactory), graph)
        
        ctx.startRecording()
        val ops = ctx.ops
        
        val a = ones(Shape(intArrayOf(1)))
        val b = ones(Shape(intArrayOf(1)))
        
        val sum = ops.add(a, b)
        val relu_result = ops.relu(sum)
        
        val tape = ctx.stopRecording() as DefaultExecutionTape
        
        // Verify both tape and graph were populated
        assertTrue(tape.traces.isNotEmpty(), "Tape should contain traces")
        assertTrue(graph.nodes.isNotEmpty(), "Graph should contain nodes")
        
        // Verify consistency between tape and graph
        val tapeOps = tape.traces.map { it.opType }.toSet()
        val graphOps = graph.nodes.map { it.operation.name }.toSet()
        assertEquals(tapeOps, graphOps, "Tape and graph should have same operations")
    }

    @Test
    fun kspTracingTensorOps_maintainsBackwardCompatibility() {
        // Test that switching from TracingTensorOps to TracingTensorOps maintains backward compatibility
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        
        // This test verifies that the same API and behavior is maintained
        ctx.startRecording()
        val ops = ctx.ops
        
        val a = ones(Shape(intArrayOf(1)))
        val b = ones(Shape(intArrayOf(1)))
        
        // Execute the same operations that worked with TracingTensorOps
        val sum = ops.add(a, b)
        val relu_result = ops.relu(sum)
        
        val tape = ctx.stopRecording()
        assertNotNull(tape)
        tape as DefaultExecutionTape
        
        // Verify the same trace structure is maintained
        val traces = tape.traces
        assertEquals(2, traces.size, "Should have 2 traces")
        
        val addTrace = traces.first { it.opType == "add" }
        val reluTrace = traces.first { it.opType == "relu" }
        
        // Verify trace structure matches expected format
        assertEquals(2, addTrace.inputs.size)
        assertEquals(1, addTrace.outputs.size)
        assertEquals(1, reluTrace.inputs.size)
        assertEquals(1, reluTrace.outputs.size)
        
        // Verify attributes are present
        assertTrue(addTrace.attributes.isNotEmpty())
        assertTrue(reluTrace.attributes.isNotEmpty())
    }

    @Test
    fun kspTracingTensorOps_handlesComplexWorkflow() {
        // Test a more complex workflow that exercises multiple aspects of TracingTensorOps
        val graph = DefaultComputeGraph()
        val ctx = DefaultGraphExecutionContext.tapeAndGraph(DefaultCpuOps(dataFactory), graph)
        
        val (tape, result) = ctx.record {
            val ops = this.ops
            
            // Create input tensors
            val input1 = ones(Shape(intArrayOf(2, 2)))
            val input2 = ones(Shape(intArrayOf(2, 2)))
            
            // Binary operations
            val sum = ops.add(input1, input2)
            val product = ops.multiply(sum, input1)
            
            // Scalar operations
            val scaled = ops.mulScalar(product, 0.5)
            
            // Activation functions
            val activated = ops.relu(scaled)
            val final = ops.sigmoid(activated)
            
            // Shape operations
            val reshaped = ops.reshape(final, Shape(intArrayOf(4)))
            
            reshaped
        }
        
        assertNotNull(tape)
        tape as DefaultExecutionTape
        
        // Verify comprehensive tracing
        val traces = tape.traces
        assertTrue(traces.size >= 6, "Should have at least 6 operation traces")
        
        val expectedOps = setOf("add", "multiply", "mulScalar", "relu", "sigmoid", "reshape")
        val actualOps = traces.map { it.opType }.toSet()
        assertTrue(expectedOps.all { it in actualOps }, "Should trace all expected operations")
        
        // Verify graph construction
        assertTrue(graph.nodes.size >= 6, "Graph should have at least 6 nodes")
        assertTrue(graph.edges.isNotEmpty(), "Graph should have edges")
        
        // Verify final result is correct
        assertNotNull(result)
        assertEquals(Shape(intArrayOf(4)), result.shape, "Final result should have correct shape")
    }
}