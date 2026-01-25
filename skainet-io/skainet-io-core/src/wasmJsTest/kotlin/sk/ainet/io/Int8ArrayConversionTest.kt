@file:OptIn(kotlin.js.ExperimentalWasmJsInterop::class)

package sk.ainet.io

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.Int8Array
import org.khronos.webgl.get

// Helper to set bytes in ArrayBuffer via JS
@JsFun("(buffer, index, value) => { new Uint8Array(buffer)[index] = value; }")
private external fun setBufferByte(buffer: ArrayBuffer, index: Int, value: Int)

// Helper to create buffer with test data
@JsFun("""(size) => {
    const buffer = new ArrayBuffer(size);
    return buffer;
}""")
private external fun createArrayBuffer(size: Int): ArrayBuffer

/**
 * Test byte array conversion in WASM environment.
 */
class Int8ArrayConversionTest {

    @Test
    fun testInt8ArrayGet() {
        // Create a typed array with known values
        val buffer = createArrayBuffer(8)

        // Set bytes: 0x08, 0x09, 0x12, 0x08 (ONNX-like protobuf header)
        setBufferByte(buffer, 0, 0x08)  // field 1, wire type 0
        setBufferByte(buffer, 1, 0x09)  // value
        setBufferByte(buffer, 2, 0x12)  // field 2, wire type 2
        setBufferByte(buffer, 3, 0x08)  // length
        setBufferByte(buffer, 4, 0xFF)  // 255 -> should become -1 as signed byte
        setBufferByte(buffer, 5, 0x80)  // 128 -> should become -128 as signed byte
        setBufferByte(buffer, 6, 0x7F)  // 127 -> should stay 127
        setBufferByte(buffer, 7, 0x00)  // 0 -> should stay 0

        val int8 = Int8Array(buffer)

        // Test reading via Int8Array
        println("Testing Int8Array.get():")
        for (i in 0 until 8) {
            val value = int8[i]
            println("  int8[$i] = $value")
        }

        // Verify values
        assertEquals(0x08.toByte(), int8[0], "byte 0")
        assertEquals(0x09.toByte(), int8[1], "byte 1")
        assertEquals(0x12.toByte(), int8[2], "byte 2")
        assertEquals(0x08.toByte(), int8[3], "byte 3")
        assertEquals((-1).toByte(), int8[4], "byte 4 (0xFF -> -1)")
        assertEquals((-128).toByte(), int8[5], "byte 5 (0x80 -> -128)")
        assertEquals(127.toByte(), int8[6], "byte 6 (0x7F -> 127)")
        assertEquals(0.toByte(), int8[7], "byte 7")
    }

    @Test
    fun testByteArrayConversion() {
        // Simulate what JsBlobRandomAccessSource does
        val buffer = createArrayBuffer(16)

        // Set some test bytes
        for (i in 0 until 16) {
            setBufferByte(buffer, i, (i * 17) and 0xFF)  // 0, 17, 34, ... wrapping at 256
        }

        val int8 = Int8Array(buffer)
        val bytes = ByteArray(int8.length) { i -> int8[i] }

        println("ByteArray conversion test:")
        for (i in 0 until 16) {
            val unsigned = (i * 17) and 0xFF
            val expected = (if (unsigned > 127) unsigned - 256 else unsigned).toByte()
            println("  bytes[$i] = ${bytes[i]}, expected = $expected")
            assertEquals(expected, bytes[i], "byte $i")
        }
    }

    @Test
    fun testOnnxMagicBytes() {
        // ONNX files start with protobuf fields
        // Field 1 (ir_version) with varint wire type: 0x08
        // Then the version number
        val buffer = createArrayBuffer(4)
        setBufferByte(buffer, 0, 0x08)  // field 1, wire type 0 (varint)
        setBufferByte(buffer, 1, 0x09)  // ir_version = 9
        setBufferByte(buffer, 2, 0x12)  // field 2, wire type 2 (length-delimited)
        setBufferByte(buffer, 3, 0x08)  // length = 8

        val int8 = Int8Array(buffer)
        val bytes = ByteArray(int8.length) { i -> int8[i] }

        println("ONNX magic bytes test:")
        println("  bytes[0] = ${bytes[0]} (expected 8)")
        println("  bytes[1] = ${bytes[1]} (expected 9)")
        println("  bytes[2] = ${bytes[2]} (expected 18)")
        println("  bytes[3] = ${bytes[3]} (expected 8)")

        // Check wire types
        val wireType0 = bytes[0].toInt() and 0x07
        val wireType2 = bytes[2].toInt() and 0x07

        println("  wireType at byte 0: $wireType0 (expected 0 = varint)")
        println("  wireType at byte 2: $wireType2 (expected 2 = length-delimited)")

        assertEquals(0, wireType0, "wire type 0 should be varint")
        assertEquals(2, wireType2, "wire type 2 should be length-delimited")

        // Wire type 3 would indicate "start group" which triggers the error
        assertTrue(wireType0 != 3, "wire type should not be 3 (group)")
        assertTrue(wireType2 != 3, "wire type should not be 3 (group)")
    }
}
