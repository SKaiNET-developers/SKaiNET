package sk.ainet.lang.nn

import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** The `gru(...)` network-DSL builder wires a [Gru] layer into the model. */
class GruDslTest {

    @Test
    fun gru_dsl_builds_a_gru_layer() {
        val model = definition<FP32, Float> {
            network {
                input(4)      // input feature size D
                gru(8)        // hidden size H
            }
        }
        assertNotNull(model)
        // The built model tree must contain a Gru module configured D=4 -> H=8.
        val grus = flattenModules(model).filterIsInstance<Gru<FP32, Float>>()
        assertTrue(grus.isNotEmpty(), "network { gru(8) } must produce a Gru module")
        assertTrue(grus.any { it.inputSize == 4 && it.hiddenSize == 8 })
    }

    private fun flattenModules(m: Module<FP32, Float>): List<Module<FP32, Float>> =
        listOf(m) + m.modules.flatMap { flattenModules(it) }
}
