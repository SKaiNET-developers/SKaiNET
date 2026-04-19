package sk.ainet.lang.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import sk.ainet.lang.types.FP32

class ConvOperationInferOutputsTest {

    private fun spec(name: String, shape: List<Int>?): TensorSpec =
        TensorSpec(name = name, shape = shape, dtype = FP32::class.simpleName!!)

    // ----- Conv1d -----

    @Test
    fun conv1d_inferOutputs_uses_weight_shape_and_params() {
        val op = Conv1dOperation<FP32, Float>(
            mapOf("stride" to 1, "padding" to 1, "dilation" to 1, "groups" to 1)
        )
        val out = op.inferOutputs(
            listOf(
                spec("input", listOf(1, 80, 3000)),
                spec("weight", listOf(384, 80, 3))
            )
        ).single()

        assertEquals(listOf(1, 384, 3000), out.shape)
        assertEquals("conv1d_output", out.name)
    }

    @Test
    fun conv1d_inferOutputs_default_params_when_missing() {
        val op = Conv1dOperation<FP32, Float>()
        val out = op.inferOutputs(
            listOf(
                spec("input", listOf(2, 3, 28)),
                spec("weight", listOf(16, 3, 3))
            )
        ).single()

        assertEquals(listOf(2, 16, 26), out.shape)
    }

    @Test
    fun conv1d_inferOutputs_returns_null_shape_when_input_unknown() {
        val op = Conv1dOperation<FP32, Float>(mapOf("stride" to 1, "padding" to 0))
        val out = op.inferOutputs(
            listOf(
                spec("input", null),
                spec("weight", listOf(16, 3, 3))
            )
        ).single()

        assertNull(out.shape)
    }

    @Test
    fun conv1d_inferOutputs_returns_null_shape_when_rank_mismatch() {
        val op = Conv1dOperation<FP32, Float>()
        val out = op.inferOutputs(
            listOf(
                spec("input", listOf(28)),
                spec("weight", listOf(16, 3, 3))
            )
        ).single()

        assertNull(out.shape)
    }

    // ----- Conv2d -----

    @Test
    fun conv2d_inferOutputs_uses_pair_params() {
        val op = Conv2dOperation<FP32, Float>(
            mapOf(
                "stride" to (2 to 2),
                "padding" to (1 to 1),
                "dilation" to (1 to 1),
                "groups" to 1
            )
        )
        val out = op.inferOutputs(
            listOf(
                spec("input", listOf(1, 3, 32, 32)),
                spec("weight", listOf(16, 3, 3, 3))
            )
        ).single()

        // (32 + 2 - 2 - 1) / 2 + 1 = 16
        assertEquals(listOf(1, 16, 16, 16), out.shape)
    }

    @Test
    fun conv2d_inferOutputs_accepts_int_param_as_symmetric() {
        val op = Conv2dOperation<FP32, Float>(
            mapOf("stride" to 1, "padding" to 0, "dilation" to 1)
        )
        val out = op.inferOutputs(
            listOf(
                spec("input", listOf(1, 3, 28, 28)),
                spec("weight", listOf(16, 3, 3, 3))
            )
        ).single()

        assertEquals(listOf(1, 16, 26, 26), out.shape)
    }

    // ----- Conv3d -----

    @Test
    fun conv3d_inferOutputs_uses_triple_params() {
        val op = Conv3dOperation<FP32, Float>(
            mapOf(
                "stride" to Triple(1, 1, 1),
                "padding" to Triple(0, 0, 0),
                "dilation" to Triple(1, 1, 1),
                "groups" to 1
            )
        )
        val out = op.inferOutputs(
            listOf(
                spec("input", listOf(1, 3, 16, 16, 16)),
                spec("weight", listOf(8, 3, 3, 3, 3))
            )
        ).single()

        assertEquals(listOf(1, 8, 14, 14, 14), out.shape)
    }

    @Test
    fun conv3d_inferOutputs_default_params_when_missing() {
        val op = Conv3dOperation<FP32, Float>()
        val out = op.inferOutputs(
            listOf(
                spec("input", listOf(1, 3, 8, 8, 8)),
                spec("weight", listOf(4, 3, 3, 3, 3))
            )
        ).single()

        assertEquals(listOf(1, 4, 6, 6, 6), out.shape)
    }
}
