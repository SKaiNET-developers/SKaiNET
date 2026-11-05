package sk.ainet.compile.hlo

import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TypeMapperDTypeTest {
    private val mapper = TypeMapper()

    @Test
    fun int4MapsToI4() {
        assertEquals("i4", mapper.mapDType("Int4"))
        assertEquals("i4", mapper.mapDType("I4"))
        assertEquals("tensor<32x64xi4>", mapper.mapTensorType(TensorSpec("w", listOf(32, 64), "Int4")))
    }

    @Test
    fun dtypeNamesFromTheLanguageMap() {
        // DType.name and simpleName spellings, as recorded by the DSL and the tape
        assertEquals("f32", mapper.mapDType("Float32"))
        assertEquals("f32", mapper.mapDType("FP32"))
        assertEquals("bf16", mapper.mapDType("BFloat16"))
        assertEquals("f16", mapper.mapDType("Float16"))
        assertEquals("i8", mapper.mapDType("Int8"))
        assertEquals("ui8", mapper.mapDType("UInt8"))
    }

    @Test
    fun unsetDtypeStaysF32() {
        assertEquals("f32", mapper.mapDType("unknown"))
        assertEquals("f32", mapper.mapDType("Unknown"))
        assertEquals("f32", mapper.mapDType(""))
    }

    @Test
    fun dtypeWithoutAnElementTypeFails() {
        val e = assertFailsWith<IllegalArgumentException> { mapper.mapDType("Ternary") }
        assertTrue("Ternary" in e.message.orEmpty())
        assertFailsWith<IllegalArgumentException> { mapper.mapDType("Q4_K") }
    }
}
