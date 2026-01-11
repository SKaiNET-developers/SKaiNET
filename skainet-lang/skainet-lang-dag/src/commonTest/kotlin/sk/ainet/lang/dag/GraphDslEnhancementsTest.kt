package sk.ainet.lang.dag

import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class GraphDslEnhancementsTest {

    @Test
    fun testTypeSafeDsl() {
        val program = dag {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 4), "Float32"))
            val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
            // Using generic op for now as matmul/add might be generated or in a different file
            // but the DSL should support typed values
            val out = op(sk.ainet.lang.tensor.ops.MatmulOperation<FP32, Float>(), listOf(x, w))
            output(out.first())
        }

        assertNotNull(program)
        assertEquals(3, program.nodes.size)
        // Check if types are preserved in metadata if possible, 
        // though GraphValue<T> is mainly for compile-time safety in DSL.
    }

    @Test
    fun testModuleSupport() {
        val linearModule = dagModule { inputs ->
            val x = inputs[0]
            val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
            val b = constant<FP32, Float>("b") { shape(4) { zeros() } }
            
            // Re-using op for demonstration
            val mm = op(sk.ainet.lang.tensor.ops.MatmulOperation<FP32, Float>(), listOf(x, w)).first()
            val out = op(sk.ainet.lang.tensor.ops.AddOperation<FP32, Float>(), listOf(mm, b))
            out
        }

        val program = dag {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 4), "Float32"))
            val y = module(linearModule, listOf(x))
            output(y.first())
        }

        assertNotNull(program)
        // input + (param + const + matmul + add) = 5 nodes
        assertEquals(5, program.nodes.size)
    }
}
