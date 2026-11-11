package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.ScopedTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * SKEEP-003 decision 10: activation ids are debug-only in eager execution. With memory debug mode
 * on, every forward-scope allocation is named `<scope>.act#step=<step>,n=<ordinal>` and recorded.
 */
class EagerActivationIdTest {

    @BeforeTest fun on() { MemoryDebug.overrideEnabled = true; MemoryDebug.reset() }

    @AfterTest fun off() { MemoryDebug.onAllocate = null; MemoryDebug.reset(); MemoryDebug.overrideEnabled = null }

    @Test
    fun withoutDebugModeActivationsAreAnonymous() {
        MemoryDebug.overrideEnabled = false
        val scope = ForwardScope(64)
        val a = scope.allocateFloats(8)
        assertEquals("forward.slab", a.debugOrigin?.canonical, "an unnamed piece of the slab")
        val data = ScopedTensorDataFactory(DenseTensorDataFactory()) { scope }.zeros<FP32, Float>(Shape(2, 2), FP32::class)
        assertNull(data.view?.id)
        scope.close()
    }

    @Test
    fun activationsAreNumberedWithinTheStepAndTheStepIsCounted() {
        val scope = ForwardScope(64, name = "decode")
        assertEquals("decode.act#step=0,n=0", scope.allocateFloats(8).debugOrigin?.canonical)
        assertEquals("decode.act#step=0,n=1", scope.allocateFloats(8).debugOrigin?.canonical)
        scope.reset()
        assertEquals("decode.act#step=1,n=0", scope.allocateFloats(8).debugOrigin?.canonical, "the same op gets the same ordinal every step")
        scope.close()
    }

    @Test
    fun overflowAllocationsAreNumberedInTheSameSequence() {
        val scope = ForwardScope(8)
        scope.allocateFloats(8)
        val overflow = scope.allocateFloats(16)
        assertEquals("forward.act#step=0,n=1", overflow.debugOrigin?.canonical)
        scope.close()
    }

    @Test
    fun anExplicitOriginWins() {
        val scope = ForwardScope(64)
        val id = TensorId.parse("model.layers[0].attn.scores")
        assertEquals(id, scope.allocateFloats(8, id).debugOrigin)
        scope.close()
    }

    @Test
    fun aTensorMadeUnderAForwardScopeCarriesItsActivationId() {
        val scope = ForwardScope(64)
        val factory = ScopedTensorDataFactory(DenseTensorDataFactory()) { scope }
        factory.zeros<FP32, Float>(Shape(2, 2), FP32::class)
        val second = factory.ones<FP32, Float>(Shape(4), FP32::class)
        assertEquals("forward.act#step=0,n=1", assertNotNull(second.view).id?.canonical)
        scope.close()
    }

    @Test
    fun aUseAfterResetNamesTheActivation() {
        val scope = ForwardScope(64)
        scope.allocateFloats(4)
        val escaped = scope.allocateFloats(8)
        scope.reset()
        val ex = assertFailsWith<StorageClosedException> { escaped.checkAlive() }
        val msg = assertNotNull(ex.message)
        assertTrue(msg.contains("forward.act#step=0,n=1"), msg)
        assertTrue(msg.contains("32 B in forward scope"), msg)
        assertTrue(!msg.contains("no debug record"), msg)
        scope.close()
    }

    @Test
    fun activationsAreInTheLedgerButDoNotInflateTheScopesBytes() {
        val scope = ForwardScope(64)
        val slabBytes = MemoryDebug.liveBytes(ScopeKind.FORWARD)
        assertEquals(256L, slabBytes)
        val seen = ArrayList<MemoryDebug.Entry>()
        MemoryDebug.onAllocate = { seen += it }
        val a = scope.allocateFloats(16)
        assertEquals(a.id, seen.single().storageId)
        assertEquals("forward.act#step=0,n=0", seen.single().origin?.canonical)
        assertEquals(slabBytes, MemoryDebug.liveBytes(ScopeKind.FORWARD), "the slab already counts these bytes")
        assertEquals(slabBytes, MemoryDebug.peakBytes(ScopeKind.FORWARD))
        assertTrue(MemoryDebug.report().contains("forward.act#step=0,n=0"), MemoryDebug.report())
        scope.reset()
        assertEquals(slabBytes, MemoryDebug.liveBytes(ScopeKind.FORWARD))
        scope.close()
    }

    @Test
    fun theLedgerKeepsOneStepOfClosedActivations() {
        val scope = ForwardScope(64)
        val perStep = ArrayList<StorageId>()
        repeat(5) {
            perStep += scope.allocateFloats(8).id
            scope.allocateFloats(8)
            scope.reset()
        }
        assertNotNull(MemoryDebug.entry(perStep[4]), "the step just reset can still be explained")
        assertTrue(assertNotNull(MemoryDebug.entry(perStep[4])).closed)
        for (step in 0..3) assertNull(MemoryDebug.entry(perStep[step]), "step $step is forgotten")
        scope.close()
    }
}
