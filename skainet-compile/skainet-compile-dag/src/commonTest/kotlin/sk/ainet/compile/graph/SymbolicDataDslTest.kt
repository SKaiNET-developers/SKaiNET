package sk.ainet.compile.graph

import sk.ainet.lang.dag.dag
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.dsl.toComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SymbolicDataDslTest {

    @Test
    fun parameterAndConstantViaSymbolicDataDslPropagateShapeDtypeAndMetadata() {
        val program = dag {
            val x = input("x", TensorSpec("x", listOf(1, 4), "FP32"))
            val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
            val b = constant<FP32, Float>("b") { fromArray(floatArrayOf(0f, 0f, 0f, 0f), shape = listOf(4)) }
            val y = add(matmul(x, w), b)
            output(y)
        }

        val graph = program.toComputeGraph() as DefaultComputeGraph

        // Parameter node should carry shape/dtype/metadata
        val paramNode = graph.nodes.first { it.id.startsWith("param_w") }
        val paramOut = paramNode.outputs.first()
        assertEquals(listOf(4, 4), paramOut.shape)
        assertEquals("FP32", paramOut.dtype)
        assertEquals("ones", paramOut.metadata["init"])

        // Constant node should carry shape/dtype and init metadata
        val constNode = graph.nodes.first { it.id.startsWith("const_b") }
        val constOut = constNode.outputs.first()
        assertEquals(listOf(4), constOut.shape)
        assertEquals("FP32", constOut.dtype)
        assertEquals("fromArray", constOut.metadata["init"])

        // Validation should succeed
        val validation = graph.validate()
        assertTrue(validation is sk.ainet.lang.tensor.ops.ValidationResult.Valid)
        assertNotNull(graph.getOutputNodes().firstOrNull { it.id.contains("add") || it.id.contains("matmul") })
    }
}
