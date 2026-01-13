package sk.ainet.lang.nn

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.topology.ModuleNode
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData

class DualModuleCompositionTest {

    private val ctx: ExecutionContext = DefaultNeuralNetworkExecutionContext()

    // A mock dual module converting Int32 data to FP32 by casting values to float
    // TODO: this functionality should be provided by SKN
    private class IntToFloatDual<V> : DualModule<Int32, FP32, V>() {
        override val name: String = "IntToFloatDual"
        override val modules: List<ModuleNode> = emptyList()
        override fun forward(input: Tensor<Int32, V>, ctx: ExecutionContext): Tensor<FP32, V> {
            val shape = input.shape
            val intData = (input.data as IntArrayTensorData<Int32>).buffer
            val floatData = FloatArray(intData.size) { idx -> intData[idx].toFloat() }
            @Suppress("UNCHECKED_CAST")
            return (ctx
                .fromFloatArray<FP32, Any>(shape, FP32::class, floatData) as Tensor<FP32, Any>) as Tensor<FP32, V>
        }
    }

    // A mock unary module that adds a scalar 1.0f to FP32 tensor
    // TODO: this functionality should be provided by SKN
    private class AddOne<V> : Module<FP32, V>() {
        override val name: String = "AddOne"
        override val modules: List<Module<FP32, V>> = emptyList()
        override fun forward(input: Tensor<FP32, V>, ctx: ExecutionContext): Tensor<FP32, V> {
            val shape = input.shape
            val floats = (input.data as FloatArrayTensorData<FP32>).buffer
            val out = FloatArray(floats.size) { i -> floats[i] + 1f }
            @Suppress("UNCHECKED_CAST")
            return (DefaultNeuralNetworkExecutionContext().fromFloatArray<FP32, Any>(shape, FP32::class, out) as Tensor<FP32, Any>) as Tensor<FP32, V>
        }
    }

    @Test
    fun testDualThenUnaryComposition() {
        val d = IntToFloatDual<Any>()
        val u = AddOne<Any>()
        val chain: DualModule<Int32, FP32, Any> = compose(d, u)
        val input: Tensor<Int32, Any> = ctx.fromIntArray<Int32, Any>(Shape(2), Int32::class, intArrayOf(1, 2)) as Tensor<Int32, Any>
        val out = chain.forward(input, ctx)
        val arr = (out.data as FloatArrayTensorData<FP32>).buffer
        assertEquals(2f, arr[0])
        assertEquals(3f, arr[1])
    }

    @Test
    fun testUnaryThenDualComposition() {
        // Create a pass-through FP32 module first, then cast to float afterwards
        val identityInt32 = object : Module<Int32, Any>() {
            override val name: String = "IdI32"
            override val modules: List<Module<Int32, Any>> = emptyList()
            override fun forward(input: Tensor<Int32, Any>, ctx: ExecutionContext): Tensor<Int32, Any> = input
        }
        val d = IntToFloatDual<Any>()

        val chain: DualModule<Int32, FP32, Any> = compose(identityInt32, d)

        val input: Tensor<Int32, Any> = ctx.fromIntArray<Int32, Any>(Shape(3), Int32::class, intArrayOf(5, -1, 0)) as Tensor<Int32, Any>
        val out = chain.forward(input, ctx)
        val arr = (out.data as FloatArrayTensorData<FP32>).buffer
        assertEquals(5f, arr[0])
        assertEquals(-1f, arr[1])
        assertEquals(0f, arr[2])
    }
}
