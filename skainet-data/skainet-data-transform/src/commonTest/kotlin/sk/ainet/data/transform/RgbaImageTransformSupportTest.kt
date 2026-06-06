/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class RgbaImageTransformSupportTest {

    private fun sampleImage2x2(): PackedRgbaImage = PackedRgbaImage(
        width = 2,
        height = 2,
        rgba = byteArrayOf(
            10, 20, 30, 0xFF.toByte(),
            40, 50, 60, 0xFF.toByte(),
            70, 80, 90, 0xFF.toByte(),
            100, 110, 120, 0xFF.toByte()
        )
    )

    @Test
    fun `crop copies requested rectangle`() {
        val cropped = cropPackedRgbaImage(sampleImage2x2(), x = 1, y = 0, width = 1, height = 2)

        assertEquals(1, cropped.width)
        assertEquals(2, cropped.height)
        assertContentEquals(
            byteArrayOf(
                40, 50, 60, 0xFF.toByte(),
                100, 110, 120, 0xFF.toByte()
            ),
            cropped.rgba
        )
    }

    @Test
    fun `pad adds border color and keeps source pixels`() {
        val padded = padPackedRgbaImage(
            image = sampleImage2x2(),
            top = 1,
            bottom = 0,
            left = 1,
            right = 0,
            red = 1,
            green = 2,
            blue = 3
        )

        assertEquals(3, padded.width)
        assertEquals(3, padded.height)
        assertContentEquals(byteArrayOf(1, 2, 3, 0xFF.toByte()), padded.rgba.copyOfRange(0, 4))
        assertContentEquals(
            byteArrayOf(10, 20, 30, 0xFF.toByte()),
            padded.rgba.copyOfRange(((1 * padded.width) + 1) * 4, ((1 * padded.width) + 1) * 4 + 4)
        )
    }

    @Test
    fun `resize nearest expands pixels predictably`() {
        val image = PackedRgbaImage(
            width = 2,
            height = 1,
            rgba = byteArrayOf(
                10, 20, 30, 0xFF.toByte(),
                40, 50, 60, 0xFF.toByte()
            )
        )

        val resized = resizePackedRgbaImage(image, width = 4, height = 1, interpolation = Interpolation.NEAREST)

        assertEquals(4, resized.width)
        assertEquals(1, resized.height)
        assertContentEquals(
            byteArrayOf(
                10, 20, 30, 0xFF.toByte(),
                10, 20, 30, 0xFF.toByte(),
                40, 50, 60, 0xFF.toByte(),
                40, 50, 60, 0xFF.toByte()
            ),
            resized.rgba
        )
    }

    @Test
    fun `rotate zero degrees keeps image unchanged`() {
        val original = sampleImage2x2()

        val rotated = rotatePackedRgbaImage(original, degrees = 0f, interpolation = Interpolation.NEAREST)

        assertEquals(original.width, rotated.width)
        assertEquals(original.height, rotated.height)
        assertContentEquals(original.rgba, rotated.rgba)
    }
}
