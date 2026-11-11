package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/** SPIKE (#1507): the `KernelKey` path must give the bits the class ladder gives. */
class Q4KKeyDispatchSpikeTest {

    @AfterTest fun off() { Q4KKeyDispatchSpike.enabled = false }

    @Test
    fun keyAndLadderAgreeBitForBit() {
        val ctx = DirectCpuExecutionContext()
        for ((batch, inputDim, outputDim) in listOf(Triple(1, 256, 8), Triple(3, 512, 16), Triple(8, 1024, 64))) {
            val blocks = (inputDim / 256) * outputDim
            val packed = ByteArray(blocks * 144).also { Random(7).nextBytes(it) }
            for (b in 0 until blocks) { packed[b * 144] = 0; packed[b * 144 + 1] = 0x3C; packed[b * 144 + 2] = 0; packed[b * 144 + 3] = 0x3C }
            val x = ctx.fromFloatArray<FP32, Float>(Shape(batch, inputDim), FP32::class, FloatArray(batch * inputDim) { ((it % 251) - 125) / 127f })
            @Suppress("UNCHECKED_CAST")
            val w = ctx.fromData(Q4_KBlockTensorData(Shape(inputDim, outputDim), packed) as TensorData<FP32, Float>, FP32::class)

            Q4KKeyDispatchSpike.enabled = false
            val ladder = ctx.ops.matmul(x, w)
            Q4KKeyDispatchSpike.enabled = true
            Q4KKeyDispatchSpike.lastKernel = null
            val key = ctx.ops.matmul(x, w)
            println("spike kernel: ${Q4KKeyDispatchSpike.lastKernel}")
            assertNotNull(Q4KKeyDispatchSpike.lastKernel, "the key path must have found and run a kernel")

            assertEquals(ladder.shape, key.shape)
            assertContentEquals(ladder.data.copyToFloatArray(), key.data.copyToFloatArray(), "batch=$batch in=$inputDim out=$outputDim")
        }
    }
}
