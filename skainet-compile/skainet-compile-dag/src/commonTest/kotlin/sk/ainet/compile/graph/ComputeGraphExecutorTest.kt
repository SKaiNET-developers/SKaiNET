package sk.ainet.compile.graph

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.graph.exec.FusedOpHandler
import sk.ainet.lang.graph.exec.LLMFusedOpHandlers
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ComputeGraphExecutorTest {

    private fun spec(name: String = "t", shape: List<Int> = listOf(1, 4)) =
        TensorSpec(name = name, shape = shape, dtype = "FP32")

    private fun inputNode(id: String) = GraphNode(
        id = id,
        operation = InputOperation<FP32, Float>(),
        inputs = emptyList(),
        outputs = listOf(spec(id))
    )

    private fun opNode(id: String, opName: String, params: Map<String, Any> = emptyMap()) = GraphNode(
        id = id,
        operation = GenericOperation(opName, parameters = params, type = "generic"),
        inputs = listOf(spec()),
        outputs = listOf(spec())
    )

    @Test
    fun executesLinearGraph() {
        // input → relu → output
        val graph = DefaultComputeGraph()
        val input = graph.addNode(inputNode("input"))
        val relu = graph.addNode(opNode("relu", "relu"))
        graph.addEdge(GraphEdge("e1", input, relu, tensorSpec = spec()))

        val executor = ComputeGraphExecutor(graph, TestTensorOps())

        val inputTensor = TestTensor(floatArrayOf(-1f, 2f, -3f, 4f))
        val results = executor.execute<FP32, Float>(mapOf("input" to inputTensor))

        assertNotNull(results["relu"])
    }

    @Test
    fun executesGraphWithTwoInputs() {
        // input_a, input_b → add → output
        val graph = DefaultComputeGraph()
        val a = graph.addNode(inputNode("input_a"))
        val b = graph.addNode(inputNode("input_b"))
        val add = graph.addNode(opNode("add", "add"))

        graph.addEdge(GraphEdge("e1", a, add, destinationInputIndex = 0, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", b, add, destinationInputIndex = 1, tensorSpec = spec()))

        val executor = ComputeGraphExecutor(graph, TestTensorOps())

        val ta = TestTensor(floatArrayOf(1f, 2f, 3f, 4f))
        val tb = TestTensor(floatArrayOf(10f, 20f, 30f, 40f))
        val results = executor.execute<FP32, Float>(mapOf("input_a" to ta, "input_b" to tb))

        assertNotNull(results["add"])
    }

    @Test
    fun executesFusedOp() {
        // Register a test fused op handler
        var handlerCalled = false
        ComputeGraphExecutor.registerFusedOp("test_fused_op", FusedOpHandler<DType, Any> { ops, inputs, params ->
            handlerCalled = true
            inputs // pass-through
        })

        val graph = DefaultComputeGraph()
        val input = graph.addNode(inputNode("input"))
        val fused = graph.addNode(opNode("fused", "test_fused_op"))
        graph.addEdge(GraphEdge("e1", input, fused, tensorSpec = spec()))

        val executor = ComputeGraphExecutor(graph, TestTensorOps())
        val results = executor.execute<FP32, Float>(mapOf("input" to TestTensor(floatArrayOf(1f))))

        assertTrue(handlerCalled, "Fused op handler should have been called")
        assertNotNull(results["fused"])
    }

    @Test
    fun registersLLMFusedHandlers() {
        // Verify that registerAll() doesn't throw
        LLMFusedOpHandlers.registerAll()
        // The handlers are now registered and would be invoked for the corresponding op names
    }

    @Test
    fun handlesTransposeFlagInMatmul() {
        // matmul with transposeB=true should work
        val graph = DefaultComputeGraph()
        val a = graph.addNode(inputNode("a"))
        val b = graph.addNode(inputNode("b"))
        val mm = graph.addNode(opNode("mm", "matmul", mapOf("transposeB" to true)))

        graph.addEdge(GraphEdge("e1", a, mm, destinationInputIndex = 0, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", b, mm, destinationInputIndex = 1, tensorSpec = spec()))

        val executor = ComputeGraphExecutor(graph, TestTensorOps())
        val results = executor.execute<FP32, Float>(
            mapOf("a" to TestTensor(floatArrayOf(1f)), "b" to TestTensor(floatArrayOf(1f)))
        )
        assertNotNull(results["mm"])
    }

    @Test
    fun emptyGraphReturnsEmptyResults() {
        val graph = DefaultComputeGraph()
        val executor = ComputeGraphExecutor(graph, TestTensorOps())
        val results = executor.execute<FP32, Float>(emptyMap())
        assertTrue(results.isEmpty())
    }
}

/**
 * Minimal test tensor that wraps a FloatArray.
 * Used for testing graph execution without needing a full tensor backend.
 */
@Suppress("UNCHECKED_CAST")
private class TestTensor(val values: FloatArray) : Tensor<FP32, Float> {
    private val tensorShape = sk.ainet.lang.tensor.Shape(intArrayOf(1, values.size))
    override val data: sk.ainet.lang.tensor.data.TensorData<FP32, Float> = object : sk.ainet.lang.tensor.data.TensorData<FP32, Float> {
        override fun get(vararg indices: Int): Float = values[indices.last()]
        override fun set(vararg indices: Int, value: Float) { values[indices.last()] = value }
        override fun copyToFloatArray(): FloatArray = values.copyOf()
        override val shape: sk.ainet.lang.tensor.Shape get() = tensorShape
    }
    override val ops: TensorOps get() = TestTensorOps()
    override val dtype: kotlin.reflect.KClass<FP32> = FP32::class
    override val gradState: sk.ainet.lang.tensor.GradState<FP32, Float> = sk.ainet.lang.tensor.GradState()
}

/**
 * Minimal TensorOps implementation for testing.
 * Returns input tensors or trivial results for each operation.
 */
@Suppress("UNCHECKED_CAST")
private class TestTensorOps : TensorOps {
    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = a
    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = a
    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = a
    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = a
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = a
    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = a
    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = a
    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = a
    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> = b
    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> = b
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = a
    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> permute(tensor: Tensor<T, V>, axes: IntArray): Tensor<T, V> = tensor
    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> leakyRelu(tensor: Tensor<T, V>, negativeSlope: Float): Tensor<T, V> = tensor
    override fun <T : DType, V> elu(tensor: Tensor<T, V>, alpha: Float): Tensor<T, V> = tensor
    override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = tensor
    override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = tensor
    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> tanh(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = tensor
    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = tensor
    override fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = tensor
    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> pow(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = a
    override fun <T : DType, V> powScalar(a: Tensor<T, V>, n: Number): Tensor<T, V> = a
    override fun <T : DType, V> log(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> log2(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> log10(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> abs(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> sign(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> clamp(tensor: Tensor<T, V>, minVal: Float, maxVal: Float): Tensor<T, V> = tensor
    override fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: sk.ainet.lang.tensor.Shape): Tensor<T, V> = tensor
    override fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> = tensor
    override fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> = tensors.first()
    override fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> = listOf(tensor)
    override fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = tensor
    override fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = tensor
    override fun <T : DType, V> narrow(tensor: Tensor<T, V>, dim: Int, start: Int, length: Int): Tensor<T, V> = tensor
    override fun <T : DType, V> pad2d(tensor: Tensor<T, V>, padLeft: Int, padRight: Int, padTop: Int, padBottom: Int): Tensor<T, V> = tensor
    override fun <T : DType, V> unfold(tensor: Tensor<T, V>, dim: Int, size: Int, step: Int): Tensor<T, V> = tensor
    override fun <T : DType, V> lt(tensor: Tensor<T, V>, value: Float): Tensor<T, V> = tensor
    override fun <T : DType, V> ge(tensor: Tensor<T, V>, value: Float): Tensor<T, V> = tensor
    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> = tensor
    override fun <TFrom : DType, TTo : DType, V> convert(tensor: Tensor<TFrom, V>, targetType: TTo): Tensor<TTo, V> = tensor as Tensor<TTo, V>
    override fun <T : DType, V> gather(input: Tensor<T, V>, indices: Tensor<DType, *>, dim: Int): Tensor<T, V> = input
    override fun <T : DType, V> indexSelect(input: Tensor<T, V>, indices: Tensor<DType, *>, dim: Int): Tensor<T, V> = input
    override fun <T : DType, V> exp(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> expm1(tensor: Tensor<T, V>): Tensor<T, V> = tensor
    override fun <T : DType, V> scaledDotProductAttention(query: Tensor<T, V>, key: Tensor<T, V>, value: Tensor<T, V>, mask: Tensor<T, V>?, scale: Float, causal: Boolean): Tensor<T, V> = query
    override fun <T : DType, V> conv1d(input: Tensor<T, V>, weight: Tensor<T, V>, bias: Tensor<T, V>?, stride: Int, padding: Int, dilation: Int, groups: Int): Tensor<T, V> = input
    override fun <T : DType, V> conv2d(input: Tensor<T, V>, weight: Tensor<T, V>, bias: Tensor<T, V>?, stride: Pair<Int, Int>, padding: Pair<Int, Int>, dilation: Pair<Int, Int>, groups: Int): Tensor<T, V> = input
    override fun <T : DType, V> conv3d(input: Tensor<T, V>, weight: Tensor<T, V>, bias: Tensor<T, V>?, stride: Triple<Int, Int, Int>, padding: Triple<Int, Int, Int>, dilation: Triple<Int, Int, Int>, groups: Int): Tensor<T, V> = input
    override fun <T : DType, V> maxPool2d(input: Tensor<T, V>, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>, padding: Pair<Int, Int>): Tensor<T, V> = input
    override fun <T : DType, V> avgPool2d(input: Tensor<T, V>, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>, padding: Pair<Int, Int>, countIncludePad: Boolean): Tensor<T, V> = input
    override fun <T : DType, V> upsample2d(input: Tensor<T, V>, scale: Pair<Int, Int>, mode: sk.ainet.lang.tensor.ops.UpsampleMode, alignCorners: Boolean): Tensor<T, V> = input
}
