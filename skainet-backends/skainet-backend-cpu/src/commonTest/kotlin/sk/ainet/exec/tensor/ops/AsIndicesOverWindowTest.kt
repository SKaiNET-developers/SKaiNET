package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.tensor.NonIntegralIndexException
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.asIndices
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertFailsWith

/**
 * `asIndices` checks float indices through the data's float window (SKEEP-003, #1507). A tensor
 * that sits inside a larger array must be checked over its own elements only: not the array's
 * first ones, and not the ones after it.
 */
class AsIndicesOverWindowTest {

    private val ctx = DirectCpuExecutionContext()

    private fun window(slab: FloatArray, from: Int, shape: Shape): Tensor<FP32, Float> =
        ctx.fromData(StorageFloatTensorData<FP32>(shape, Storage.Heap.wrap(slab).slice(from * 4L, shape.volume * 4L)), FP32::class)

    @Test
    fun onlyTheWindowsOwnElementsAreChecked() {
        // Non-integral values before and after the window, integral ones inside it.
        val slab = floatArrayOf(0.5f, 0.5f, 3f, 0f, 7f, 2f, 0.5f)
        window(slab, 2, Shape(2, 2)).asIndices()
    }

    @Test
    fun aNonIntegralValueInsideTheWindowIsFound() {
        val slab = floatArrayOf(1f, 1f, 3f, 0f, 7.25f, 2f, 1f)
        assertFailsWith<NonIntegralIndexException> { window(slab, 2, Shape(2, 2)).asIndices() }
    }
}
