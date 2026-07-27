package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.ops.VoidTensorOps
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The first-class dynamic-dimension vocabulary: [Dim.DYNAMIC] is a reserved sentinel distinct from
 * reshape's `-1` = infer, dynamic-aware shape arithmetic lives in [Dim], and [Shape] / the slice DSL /
 * the shape-only tracer all treat a dynamic extent explicitly instead of doing integer math on `-1`.
 */
class DimTest {

    @Test
    fun dynamic_sentinel_is_distinct_from_reshape_infer() {
        assertTrue(Dim.isDynamic(Dim.DYNAMIC))
        assertFalse(Dim.isDynamic(-1), "reshape's -1 (infer) must NOT read as dynamic")
        assertFalse(Dim.isDynamic(0))
        assertTrue(Dim.isStatic(0))
        assertFalse(Dim.isStatic(Dim.DYNAMIC))
    }

    @Test
    fun dim_arithmetic_keeps_dynamic() {
        assertEquals(6, Dim.concat(listOf(2, 4)))
        assertEquals(Dim.DYNAMIC, Dim.concat(listOf(Dim.DYNAMIC, 1)), "? ++ 1 must stay ? (not 0)")
        assertEquals(Dim.DYNAMIC, Dim.concat(listOf(3, Dim.DYNAMIC, 5)))
        assertTrue(Dim.compatible(Dim.DYNAMIC, 7), "dynamic is compatible with any concrete size")
        assertTrue(Dim.compatible(7, 7))
        assertFalse(Dim.compatible(7, 8))
        assertEquals("?", Dim.render(Dim.DYNAMIC))
        assertEquals("40", Dim.render(40))
    }

    @Test
    fun shape_reports_dynamic_axes_and_guards_volume() {
        val s = Shape(1, 8, Dim.DYNAMIC, 40)
        assertTrue(s.hasDynamic())
        assertTrue(s.isDynamic(2))
        assertFalse(s.isDynamic(1))
        assertEquals(listOf(2), s.dynamicAxes)
        assertTrue(s.toString().contains("?"))
        // volume is undefined for a dynamic shape — must throw rather than return a corrupt product.
        assertFailsWith<IllegalArgumentException> { s.volume }
        // static shapes are unaffected.
        assertEquals(1 * 8 * 5 * 40, Shape(1, 8, 5, 40).volume)
    }

    @Test
    fun void_concat_keeps_growing_cache_dynamic() {
        val ops = VoidTensorOps()
        val past = VoidOpsTensor<sk.ainet.lang.types.FP32, Float>(
            ShapeOnly(Shape(1, 4, Dim.DYNAMIC, 256)), sk.ainet.lang.types.FP32::class,
        )
        val step = VoidOpsTensor<sk.ainet.lang.types.FP32, Float>(
            ShapeOnly(Shape(1, 4, 1, 256)), sk.ainet.lang.types.FP32::class,
        )
        val cat = ops.concat(listOf(past, step), dim = 2)
        assertEquals(listOf(1, 4, Dim.DYNAMIC, 256), cat.shape.dimensions.toList(), "past ++ step keeps the seq axis dynamic")
    }

    @Test
    fun slice_all_is_symbolic_full_axis_over_dynamic() {
        val all = Slice.All<sk.ainet.lang.types.FP32, Float>()
        assertTrue(all.isValid(Dim.DYNAMIC))
        assertEquals(Dim.DYNAMIC, all.getResultSize(Dim.DYNAMIC), "all() over a dynamic axis stays dynamic")
        // A partial range with concrete non-negative bounds over a dynamic axis yields a concrete size.
        val r = Slice.Range<sk.ainet.lang.types.FP32, Float>(2, 5)
        assertTrue(r.isValid(Dim.DYNAMIC))
        assertEquals(3, r.getResultSize(Dim.DYNAMIC))
    }
}

/** Minimal shape-only TensorData for the concat test (mirrors VoidTensorOps' internal one). */
private class ShapeOnly<T : sk.ainet.lang.types.DType, V>(
    override val shape: Shape,
) : sk.ainet.lang.tensor.data.TensorData<T, V> {
    override fun get(vararg indices: Int): V = error("shape-only")
    override fun set(vararg indices: Int, value: V) = error("shape-only")
    override fun copyToFloatArray(): FloatArray = error("shape-only")
}
