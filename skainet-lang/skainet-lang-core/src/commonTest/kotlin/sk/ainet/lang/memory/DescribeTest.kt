package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DescribeTest {

    @Test
    fun tensorRendererPrintsIdFormatShapeKindAndPlaceholders() {
        val t = VoidOpsTensor(DenseFloatArrayTensorData<FP32>(Shape(2, 3), FloatArray(6)), FP32::class)
        val anon = t.describe().split(" · ")
        assertEquals(7, anon.size)
        assertEquals("—", anon[0]); assertEquals("Float32/Dense(4B)", anon[1]); assertEquals("[2, 3]", anon[2])
        assertEquals("—", anon[4]); assertEquals("scope —", anon[5]); assertEquals("storage —", anon[6])

        t.id = TensorId(listOf("model", "layers", "blk.3", "attn"), "q_proj.weight")
        assertTrue(t.describe().startsWith("model.layers.blk.3.attn.q_proj.weight · Float32/Dense(4B) · [2, 3] · "))
    }

    @Test
    fun packedWeightRendersLogicalDtypeAndEncoding() {
        @Suppress("UNCHECKED_CAST")
        val q = VoidOpsTensor(Q4_KBlockTensorData(Shape(1, 256), ByteArray(144)) as TensorData<FP32, Float>, FP32::class)
        assertTrue(q.describe().contains(" · Float32/Q4_K · [1, 256] · "), q.describe())
    }

    @Test
    fun viewRendererShowsStorageKindOwnerScopeAndId() {
        val id = TensorId.parse("model.layers.blk.3.attn.q_proj.weight")
        val packed = TensorView.ofBytes(
            Storage.Heap.wrap(ByteArray(144), mutable = false), Shape(1, 256), Format(FP32, TensorEncoding.Q4_K), id,
        )
        val parts = packed.describe().split(" · ")
        assertEquals(7, parts.size)
        assertEquals("model.layers.blk.3.attn.q_proj.weight", parts[0])
        assertEquals("Float32/Q4_K", parts[1]); assertEquals("[1, 256]", parts[2])
        assertEquals("Heap/Borrowed", parts[3]); assertEquals("—", parts[4])
        assertEquals("scope ${packed.storage.scope.name}", parts[5])
        assertEquals("storage ${packed.storage.id}", parts[6])

        val owned = TensorView.dense(Storage.Heap.floats(4), Shape(4))
        assertTrue(owned.describe().startsWith("— · Float32/Dense(4B) · [4] · Heap/Owned · — · scope AMBIENT · storage "), owned.describe())
    }
}
