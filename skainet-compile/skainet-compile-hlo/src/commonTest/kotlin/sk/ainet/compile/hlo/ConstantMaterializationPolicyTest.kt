package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * End-to-end tests for the [ConstantMaterializationPolicy] seam
 * introduced for issue #523. Covers the three policies through the
 * full [StableHloConverterFactory.createBasic] pipeline so that every
 * handoff (converter → context → module → output) is exercised.
 */
class ConstantMaterializationPolicyTest {

    @Test
    fun testDefaultPolicyIsInlineAlways() {
        // Callers that construct a converter without naming a policy
        // must see the historical inline emission — the seam only
        // activates when explicitly opted into.
        val module = StableHloConverterFactory.createBasic()
            .convert(buildTensorConstantGraph(), "default_policy")

        assertTrue(module.content.contains("stablehlo.constant"))
        assertTrue(module.externalParameters.isEmpty())
        assertFalse(module.content.contains("util.global"))
    }

    @Test
    fun testExternalAlwaysEmitsUtilGlobalAndRegistersRef() {
        // With ExternalAlways the converter must:
        //  - emit `util.global private @<key> : <type>` at module scope,
        //  - emit `util.global.load @<key>` instead of inline dense<>,
        //  - register a matching ExternalParameterRef on the module.
        val module = StableHloConverterFactory.createBasic(
            ConstantMaterializationPolicy.ExternalAlways()
        ).convert(buildTensorConstantGraph(), "external_policy")

        // The util.global must carry a #flow.parameter.named initializer
        // so iree-compile binds the declaration to an archive entry at
        // --iree-opt-import-parameters time (see issue #523).
        assertTrue(
            module.content.contains(
                "util.global private @weights = " +
                    "#flow.parameter.named<\"model\"::\"weights\"> : tensor<2x2xf32>"
            ),
            "module decl missing:\n${module.content}"
        )
        assertTrue(
            module.content.contains("util.global.load @weights : tensor<2x2xf32>"),
            "util.global.load missing:\n${module.content}"
        )
        // No inline dense<> for the externalized constant — only the
        // `module attributes` clause on the header is permitted to
        // mention the tensor name. Nothing in the function body
        // should spell values out.
        assertFalse(
            module.content.contains("stablehlo.constant dense<"),
            "externalized tensor leaked an inline stablehlo.constant:\n${module.content}"
        )

        assertEquals(1, module.externalParameters.size)
        val ref = module.externalParameters.single()
        assertEquals("model", ref.scope)
        assertEquals("weights", ref.key)
        // 4 f32 elements = 16 bytes
        assertEquals(16L, ref.source.sizeInBytes)
    }

    @Test
    fun testSizeThresholdSplitsBySize() {
        // A 2x2 f32 tensor is 16 bytes; threshold of 32 bytes must
        // keep it inline. Raise the stakes by also including a 4x4
        // f32 tensor (64 bytes) which should be externalized.
        val graph = DefaultComputeGraph()
        graph.addNode(
            GraphNode(
                id = "small",
                operation = mockConstantOp(
                    "tensor_constant",
                    mapOf("values" to listOf(1.0f, 2.0f, 3.0f, 4.0f))
                ),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("small_w", listOf(2, 2), "FP32"))
            )
        )
        graph.addNode(
            GraphNode(
                id = "large",
                operation = mockConstantOp(
                    "tensor_constant",
                    mapOf("values" to List(16) { 0.0f })
                ),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("large_w", listOf(4, 4), "FP32"))
            )
        )

        val module = StableHloConverterFactory.createBasic(
            ConstantMaterializationPolicy.SizeThreshold(bytes = 32L)
        ).convert(graph, "threshold")

        // Small stays inline — `stablehlo.constant dense<...>` shows up
        // for the 2x2 tensor.
        assertTrue(
            module.content.contains("stablehlo.constant dense<"),
            "small tensor must remain inline under SizeThreshold:\n${module.content}"
        )
        // Large is externalized.
        assertTrue(
            module.content.contains(
                "util.global private @large_w = " +
                    "#flow.parameter.named<\"model\"::\"large_w\"> : tensor<4x4xf32>"
            ),
            "large tensor must externalize under SizeThreshold:\n${module.content}"
        )

        // Exactly one ref — the large one.
        assertEquals(1, module.externalParameters.size)
        assertEquals("large_w", module.externalParameters.single().key)
    }

    @Test
    fun testModuleAttrsHeaderStillEmittedAboveUtilGlobal() {
        // When a tensor carries a tensorEncoding we already emit a
        // `module attributes { skainet.tensor_encodings = {...} } {`
        // header. The new util.global decls must slot in AFTER that
        // header, before func.func — otherwise IREE's parser chokes.
        // This test is aspirational for now: we only assert both
        // lines exist in the expected relative order.
        val graph = DefaultComputeGraph()
        graph.addNode(
            GraphNode(
                id = "w",
                operation = mockConstantOp(
                    "tensor_constant",
                    mapOf("values" to listOf(1.0f, 2.0f, 3.0f, 4.0f))
                ),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("wkey", listOf(2, 2), "FP32"))
            )
        )
        val module = StableHloConverterFactory.createBasic(
            ConstantMaterializationPolicy.ExternalAlways()
        ).convert(graph, "ordering")

        val headerIdx = module.content.indexOf("module {")
        val utilIdx = module.content.indexOf("util.global private")
        val funcIdx = module.content.indexOf("func.func")
        assertTrue(headerIdx >= 0, "module { missing")
        assertTrue(utilIdx > headerIdx, "util.global must follow module header")
        assertTrue(funcIdx > utilIdx, "func.func must follow util.global decls")
    }

    private fun buildTensorConstantGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        graph.addNode(
            GraphNode(
                id = "input",
                operation = InputOperation<DType, Any>(),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("x", listOf(2, 2), "FP32"))
            )
        )
        graph.addNode(
            GraphNode(
                id = "w",
                operation = mockConstantOp(
                    "tensor_constant",
                    mapOf("values" to listOf(1.0f, 2.0f, 3.0f, 4.0f))
                ),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("weights", listOf(2, 2), "FP32"))
            )
        )
        return graph
    }

    private fun mockConstantOp(name: String, parameters: Map<String, Any>): Operation =
        object : Operation {
            override val name: String = name
            override val type: String = "constant"
            override val parameters: Map<String, Any> = parameters

            override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> {
                throw UnsupportedOperationException("mock")
            }

            override fun validateInputs(inputs: List<TensorSpec>): ValidationResult =
                ValidationResult.Valid

            override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()

            override fun clone(newParameters: Map<String, Any>): Operation = this

            override fun serialize(): Map<String, Any> =
                mapOf("name" to name, "type" to type, "parameters" to parameters)
        }
}
