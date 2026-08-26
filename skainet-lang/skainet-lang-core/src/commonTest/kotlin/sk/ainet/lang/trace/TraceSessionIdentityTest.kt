package sk.ainet.lang.trace

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * #1178: the tape captures what only it can see — the live tensor's storage encoding as an
 * object (block size intact) and, when someone who knows it registered one, its module-path
 * identity. Refs are immutable and cached, so identity must be registered before first use.
 */
class TraceSessionIdentityTest {

    private val ctx = DefaultDataExecutionContext()

    @Test
    fun refCapturesThePackedEncodingObject() {
        val packed = Q8_0BlockTensorData.fromRawBytes(Shape(32), ByteArray(34))
        @Suppress("UNCHECKED_CAST")
        val tensor = ctx.fromData(packed as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, FP32::class)
        val ref = TraceSession().refOf(tensor)
        assertEquals(TensorEncoding.Q8_0, ref.encoding, "the encoding object, not a display name")
    }

    @Test
    fun denseTensorHasNoEncodingAndNoIdentityByDefault() {
        val tensor = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val ref = TraceSession().refOf(tensor)
        assertNull(ref.encoding)
        assertNull(ref.tensorId)
    }

    @Test
    fun identityRegisteredBeforeFirstUseIsCarried() {
        val session = TraceSession()
        val weight = ctx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, FloatArray(4))
        val id = TensorId(listOf("model", "layers[3]", "attn"), "q_proj.weight")
        session.identify(weight, id)
        assertEquals(id, session.refOf(weight).tensorId)
    }

    @Test
    fun identityRegisteredAfterFirstUseDoesNotRetrofitTheCachedRef() {
        val session = TraceSession()
        val weight = ctx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, FloatArray(2))
        val before = session.refOf(weight)
        assertNull(before.tensorId)
        session.identify(weight, TensorId(listOf("model"), "weight"))
        assertNull(session.refOf(weight).tensorId, "refs are immutable and cached — identify before first refOf")
    }
}
