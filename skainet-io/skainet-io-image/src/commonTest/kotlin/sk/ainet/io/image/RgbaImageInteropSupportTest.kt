/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.io.image

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP16
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class RgbaImageInteropSupportTest {

    private val ctx = DefaultDataExecutionContext()

    @Test
    fun `packed rgba converts to rgb chw tensor`() {
        val tensor = packedRgbaToTensor(
            PackedRgbaImage(
                width = 2,
                height = 1,
                rgba = byteArrayOf(
                    10, 20, 30, 0xFF.toByte(),
                    40, 50, 60, 0xFF.toByte()
                )
            ),
            ctx
        )

        assertEquals(Shape(1, 3, 1, 2), tensor.shape)
        assertEquals(10f, tensor.data[0, 0, 0, 0])
        assertEquals(40f, tensor.data[0, 0, 0, 1])
        assertEquals(20f, tensor.data[0, 1, 0, 0])
        assertEquals(50f, tensor.data[0, 1, 0, 1])
        assertEquals(30f, tensor.data[0, 2, 0, 0])
        assertEquals(60f, tensor.data[0, 2, 0, 1])
    }

    @Test
    fun `rgb tensor converts to opaque rgba`() {
        val tensor = ctx.fromFloatArray<FP16, Float>(
            shape = Shape(1, 3, 1, 2),
            dtype = FP16::class,
            data = floatArrayOf(
                10f, 40f,
                20f, 50f,
                30f, 60f
            )
        )

        val rgba = tensorToPackedRgba(tensor)

        assertEquals(2, rgba.width)
        assertEquals(1, rgba.height)
        assertContentEquals(
            byteArrayOf(
                10, 20, 30, 0xFF.toByte(),
                40, 50, 60, 0xFF.toByte()
            ),
            rgba.rgba
        )
    }

    @Test
    fun `single channel tensor expands to grayscale rgba`() {
        val tensor = ctx.fromFloatArray<FP16, Float>(
            shape = Shape(1, 1, 1, 2),
            dtype = FP16::class,
            data = floatArrayOf(12f, 34f)
        )

        val rgba = tensorToPackedRgba(tensor)

        assertContentEquals(
            byteArrayOf(
                12, 12, 12, 0xFF.toByte(),
                34, 34, 34, 0xFF.toByte()
            ),
            rgba.rgba
        )
    }

    @Test
    fun `rgb byte array drops alpha channel`() {
        val rgb = rgbByteArrayFromPackedRgba(
            PackedRgbaImage(
                width = 2,
                height = 1,
                rgba = byteArrayOf(
                    10, 20, 30, 0x7F,
                    40, 50, 60, 0x00
                )
            )
        )

        assertContentEquals(byteArrayOf(10, 20, 30, 40, 50, 60), rgb)
    }
}
