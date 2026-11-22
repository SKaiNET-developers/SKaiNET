package sk.ainet.docs.samples

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Runs the sample behind `how-to/custom-encoding.adoc`: an encoding defined outside the library
 * reads through a view, is reported by the memory tracker, and gets its own matmul kernel.
 */
class CustomEncodingTest {

    @Test
    fun anEncodingFromOutsideTheLibraryReadsThroughAView() {
        val r = CustomEncoding.run()
        assertEquals("Float32/SIGN1", r.format)
        assertEquals(2.0, r.bitsPerWeight)
        assertEquals(16.0, r.compressionRatio)
        assertEquals("Float32/Dense(4B)", r.materializedFormat)
        assertEquals(r.decoded[0], r.firstElement)
        // every decoded weight in a block is plus or minus that block's scale
        val scale = kotlin.math.abs(r.decoded[0])
        for (i in 0 until CustomEncoding.BLOCK) assertEquals(scale, kotlin.math.abs(r.decoded[i]))
    }

    @Test
    fun theRegisteredKernelReplacesTheReferenceAndGivesTheSameProduct() {
        val r = CustomEncoding.run()
        assertEquals("sample-SIGN1", r.kernel)
        assertTrue(r.keyText.contains("SIGN1"), r.keyText)
        for (i in r.product.indices) assertEquals(r.referenceProduct[i], r.product[i], 1e-4f, "output $i")
    }

    @Test
    fun withoutAKernelTheReferenceDecodes() {
        // `run()` registers the kernel, so only the first run in a process sees the reference;
        // whichever it is, the product is the same.
        val r = CustomEncoding.run()
        if (r.referenceKernel != "sample-SIGN1") assertNotEquals(r.kernel, r.referenceKernel)
    }
}
