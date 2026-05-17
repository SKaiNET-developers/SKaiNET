package sk.ainet.lang.dag

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import sk.ainet.lang.tensor.ops.MatmulOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32

class DtypePolicyDslTest {

    @Test
    fun op_with_dtypePolicy_records_policy_under_known_attribute_key() {
        val program = dag {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 4), "Float32"))
            val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
            op(
                operation = MatmulOperation<FP32, Float>(),
                inputs = listOf(x, w),
                dtypePolicy = DTypePolicy.Require(BF16),
            )
        }

        val mmNode = program.nodes.last { it.operation is MatmulOperation<*, *> }
        val policy = mmNode.dtypePolicy()
        assertNotNull(policy, "node must carry the DTypePolicy attribute")
        assertEquals(DTypePolicy.Require(BF16), policy)
        // The attribute lands under the shared constant key so the
        // constraint-resolution pass can find it.
        assertEquals(policy, mmNode.attributes[DTYPE_POLICY_ATTRIBUTE_KEY])
    }

    @Test
    fun nodes_without_dtypePolicy_return_null_from_accessor() {
        val program = dag {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 4), "Float32"))
            val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
            op(MatmulOperation<FP32, Float>(), listOf(x, w))  // no dtypePolicy
        }
        val mmNode = program.nodes.last { it.operation is MatmulOperation<*, *> }
        assertNull(mmNode.dtypePolicy(), "absent policy must return null, not throw")
    }

    @Test
    fun extraAttributes_preserved_alongside_dtypePolicy() {
        val program = dag {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 4), "Float32"))
            val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
            op(
                operation = MatmulOperation<FP32, Float>(),
                inputs = listOf(x, w),
                dtypePolicy = DTypePolicy.Prefer(BF16),
                extraAttributes = mapOf("note" to "attention projection"),
            )
        }
        val mmNode = program.nodes.last { it.operation is MatmulOperation<*, *> }
        assertEquals(DTypePolicy.Prefer(BF16), mmNode.dtypePolicy())
        assertEquals("attention projection", mmNode.attributes["note"])
    }
}
