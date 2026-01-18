/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

/**
 * Tests for the core Transform interface and composition.
 */
class TransformTest {

    // Simple test transforms for unit testing
    private class AddOne : Transform<Int, Int> {
        override fun apply(input: Int): Int = input + 1
        override fun getOutputShape(inputShape: Shape): Shape = inputShape
    }

    private class MultiplyTwo : Transform<Int, Int> {
        override fun apply(input: Int): Int = input * 2
        override fun getOutputShape(inputShape: Shape): Shape = inputShape
    }

    private class IntToString : Transform<Int, String> {
        override fun apply(input: Int): String = input.toString()
        override fun getOutputShape(inputShape: Shape): Shape = inputShape
    }

    private class StringLength : Transform<String, Int> {
        override fun apply(input: String): Int = input.length
        override fun getOutputShape(inputShape: Shape): Shape = Shape(1)
    }

    @Test
    fun `identity transform returns input unchanged`() {
        val identity = Identity<Int>()
        assertEquals(42, identity.apply(42))
        assertEquals(0, identity.apply(0))
        assertEquals(-1, identity.apply(-1))
    }

    @Test
    fun `identity preserves shape`() {
        val identity = Identity<Float>()
        val shape = Shape(2, 3, 4)
        assertEquals(shape, identity.getOutputShape(shape))
    }

    @Test
    fun `pipeline function creates identity`() {
        val p = pipeline<Int>()
        assertIs<Identity<Int>>(p)
        assertEquals(100, p.apply(100))
    }

    @Test
    fun `then chains transforms correctly`() {
        val addOne = AddOne()
        val multiplyTwo = MultiplyTwo()

        val pipeline = addOne then multiplyTwo

        // (5 + 1) * 2 = 12
        assertEquals(12, pipeline.apply(5))
        // (0 + 1) * 2 = 2
        assertEquals(2, pipeline.apply(0))
    }

    @Test
    fun `chained transforms can change types`() {
        val addOne = AddOne()
        val intToString = IntToString()

        val pipeline = addOne then intToString

        assertEquals("6", pipeline.apply(5))
        assertEquals("1", pipeline.apply(0))
        assertEquals("100", pipeline.apply(99))
    }

    @Test
    fun `multiple transforms can be chained`() {
        val pipeline = AddOne() then MultiplyTwo() then AddOne() then MultiplyTwo()

        // ((5 + 1) * 2 + 1) * 2 = (12 + 1) * 2 = 26
        assertEquals(26, pipeline.apply(5))
    }

    @Test
    fun `pipeline with identity at start works correctly`() {
        val pipeline = pipeline<Int>() then AddOne() then MultiplyTwo()

        // (10 + 1) * 2 = 22
        assertEquals(22, pipeline.apply(10))
    }

    @Test
    fun `shape propagates through pipeline`() {
        val stringLength = StringLength()
        val intToString = IntToString()

        // This changes shape from input to Shape(1)
        val pipeline = intToString then stringLength

        val inputShape = Shape(10)
        val outputShape = pipeline.getOutputShape(inputShape)

        assertEquals(Shape(1), outputShape)
    }

    @Test
    fun `chained transform toString is descriptive`() {
        val pipeline = AddOne() then MultiplyTwo()
        val str = pipeline.toString()

        // Should contain "ChainedTransform"
        assertEquals(true, str.contains("ChainedTransform"))
    }

    @Test
    fun `identity toString is descriptive`() {
        val identity = Identity<Int>()
        assertEquals("Identity", identity.toString())
    }

    @Test
    fun `complex pipeline preserves associativity`() {
        val a = AddOne()
        val b = MultiplyTwo()
        val c = AddOne()

        // (a then b) then c should equal a then (b then c)
        val leftAssoc = (a then b) then c
        val rightAssoc = a then (b then c)

        // Both should give the same result for any input
        for (input in listOf(0, 1, 5, 10, -1)) {
            assertEquals(
                leftAssoc.apply(input),
                rightAssoc.apply(input),
                "Associativity failed for input $input"
            )
        }
    }
}
