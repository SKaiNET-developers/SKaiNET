package sk.ainet.compile.graph

import sk.ainet.lang.dag.dag
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.dsl.toComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Smoke-test a YOLOv8-like head expressed with the DAG DSL:
 * stem conv -> downsample conv -> upsample -> 1x1 head conv, with multiple parameter/constant nodes.
 */
class YoloStyleGraphDslTest {

    @Test
    fun yoloish_head_graph_builds_and_validates() {
        val program = dag {
            val input = input("input", TensorSpec("input", listOf(1, 3, 640, 640), "FP32"))

            val w1 = parameter<FP32, Float>("w1") { shape(16, 3, 3, 3) { ones() } }
            val b1 = constant<FP32, Float>("b1") { shape(16) { zeros() } }
            val c1 = conv2d(input, w1, b1, stride = 2 to 2, padding = 1 to 1)

            val w2 = parameter<FP32, Float>("w2") { shape(32, 16, 3, 3) { ones() } }
            val b2 = constant<FP32, Float>("b2") { shape(32) { zeros() } }
            val c2 = conv2d(c1, w2, b2, stride = 2 to 2, padding = 1 to 1)

            val up = upsample2d(c2, scale = 2 to 2, mode = "nearest")

            val wHead = parameter<FP32, Float>("w_head") { shape(3, 32, 1, 1) { ones() } }
            val bHead = constant<FP32, Float>("b_head") { shape(3) { zeros() } }
            val head = conv2d(up, wHead, bHead, stride = 1 to 1, padding = 0 to 0)

            output(c2, head) // multi-scale outputs like YOLO heads
        }

        val graph = program.toComputeGraph() as DefaultComputeGraph

        // Node and edge counts (6 param/const + 4 ops + 1 input = 11 nodes; 10 edges from wiring)
        assertEquals(11, graph.nodes.size)
        assertEquals(10, graph.edges.size)

        val opNames = graph.nodes.map { it.operation.name }.toSet()
        assertTrue(opNames.contains("conv2d"))
        assertTrue(opNames.contains("upsample2d"))
        assertTrue(opNames.contains("input"))

        val validation = graph.validate()
        assertTrue(validation is sk.ainet.lang.tensor.ops.ValidationResult.Valid, "Graph should validate: $validation")

        // Ensure head stays as an output node (no outgoing edges)
        val outputNodes = graph.getOutputNodes().map { it.id }
        assertEquals(1, outputNodes.size, "Head conv should be the sole output node")
        assertTrue(outputNodes.first().contains("conv2d"))
    }
}
