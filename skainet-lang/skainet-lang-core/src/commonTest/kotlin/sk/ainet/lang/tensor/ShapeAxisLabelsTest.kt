package sk.ainet.lang.tensor

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

/** SKEEP-003 decision 8: optional axis labels on [Shape], additive. */
class ShapeAxisLabelsTest {

    @Test
    fun aShapeHasNoLabelsUnlessAsked() {
        val shape = Shape(2, 16, 64)
        assertNull(shape.axisLabels)
        assertNull(shape.axisOfOrNull("seq"))
        assertFailsWith<IllegalArgumentException> { shape["seq"] }
        assertEquals("Shape: Dimensions = [2 x 16 x 64], Size (Volume) = 2048", shape.toString())
    }

    @Test
    fun labelsNameAxesAndReadExtents() {
        val shape = Shape(2, 16, 64).withAxisLabels("batch", "seq", "dim")
        assertEquals(listOf("batch", "seq", "dim"), shape.axisLabels)
        assertEquals(1, shape.axisOf("seq"))
        assertEquals(16, shape["seq"])
        assertEquals(64, shape["dim"])
        assertEquals(2, shape[0])
        assertNull(shape.axisOfOrNull("head"))
        assertFailsWith<IllegalArgumentException> { shape.axisOf("head") }
        assertEquals("Shape: Dimensions = [batch: 2 x seq: 16 x dim: 64], Size (Volume) = 2048", shape.toString())
    }

    @Test
    fun labeledBuildsExtentsAndNamesTogether() {
        val shape = Shape.labeled("batch" to 4, "head" to 8, "seq" to 32)
        assertEquals(Shape(4, 8, 32), shape)
        assertEquals(8, shape["head"])
    }

    @Test
    fun labelsAreNotPartOfTheShapesIdentity() {
        val plain = Shape(2, 3)
        val labeled = plain.withAxisLabels("rows", "cols")
        assertEquals(plain, labeled)
        assertEquals(plain.hashCode(), labeled.hashCode())
        assertEquals(plain, labeled.withAxisLabels("a", "b"))
        assertNull(plain.axisLabels, "labelling returns a new shape and leaves the original alone")
        assertNull(labeled.copy().axisLabels, "copy() is about the extents")
    }

    @Test
    fun labelsMustMatchTheRankAndBeDistinctAndNonBlank() {
        val shape = Shape(2, 3)
        assertFailsWith<IllegalArgumentException> { shape.withAxisLabels("rows") }
        assertFailsWith<IllegalArgumentException> { shape.withAxisLabels("rows", "rows") }
        assertFailsWith<IllegalArgumentException> { shape.withAxisLabels("rows", " ") }
    }

    @Test
    fun aDynamicAxisKeepsItsLabel() {
        val shape = Shape.labeled("batch" to Dim.DYNAMIC, "dim" to 8)
        assertEquals("Shape: Dimensions = [batch: ? x dim: 8], Size (Volume) = dynamic", shape.toString())
    }
}
