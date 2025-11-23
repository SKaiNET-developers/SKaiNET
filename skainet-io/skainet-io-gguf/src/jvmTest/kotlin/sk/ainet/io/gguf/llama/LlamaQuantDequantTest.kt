package sk.ainet.io.gguf.llama

import org.junit.Test
import kotlin.test.assertContentEquals

class LlamaQuantDequantTest {

    @Test
    fun `dequant Q4_0 block with scale 1 and zero codes`() {
        // d = 1.0 (0x3C00 little endian), qs = 0x88 repeated -> (8-8)=0
        val raw = ByteArray(2 + 16) { idx ->
            when (idx) {
                0 -> 0x00
                1 -> 0x3C
                else -> 0x88.toByte()
            }
        }.toList()
        val out = LlamaWeightLoader.dequantQ4_0(raw, 32)
        assertContentEquals(FloatArray(32) { 0f }.toList(), out.toList())
    }

    @Test
    fun `dequant Q8_0 block with scale 1 and ascending codes`() {
        val raw = ByteArray(2 + 32) { idx ->
            when (idx) {
                0 -> 0x00
                1 -> 0x3C
                else -> (idx - 1).toByte() // 1..32
            }
        }.toList()
        val out = LlamaWeightLoader.dequantQ8_0(raw, 32)
        val expected = FloatArray(32) { (it + 1).toFloat() }
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q5_0 block with high bits set and zero low codes yields zeros`() {
        val raw = ByteArray(2 + 4 + 16) { idx ->
            when (idx) {
                0 -> 0x00
                1 -> 0x3C // d = 1
                in 2..5 -> 0xFF.toByte() // qh all ones, but low nibble = 0 so value = 16-16=0
                else -> 0x00
            }
        }.toList()
        val out = LlamaWeightLoader.dequantQ5_0(raw, 32)
        assertContentEquals(FloatArray(32) { 0f }.toList(), out.toList())
    }

    @Test
    fun `dequant Q4_1 returns min when codes are zero`() {
        // d=1, m=2 -> bytes: d(0x00 0x3C), m(0x00 0x40), qs zeros
        val raw = ByteArray(4 + 16) { idx ->
            when (idx) {
                0 -> 0x00; 1 -> 0x3C; 2 -> 0x00; 3 -> 0x40
                else -> 0x00
            }
        }.toList()
        val out = LlamaWeightLoader.dequantQ4_1(raw, 32)
        assertContentEquals(FloatArray(32) { 2f }.toList(), out.toList())
    }

    @Test
    fun `dequant Q8_1 returns min when codes are zero`() {
        // d=1, m=2 -> bytes: d(0x00 0x3C), m(0x00 0x40), qs zeros
        val raw = ByteArray(4 + 32) { idx ->
            when (idx) {
                0 -> 0x00; 1 -> 0x3C; 2 -> 0x00; 3 -> 0x40
                else -> 0x00
            }
        }.toList()
        val out = LlamaWeightLoader.dequantQ8_1(raw, 32)
        assertContentEquals(FloatArray(32) { 2f }.toList(), out.toList())
    }

    @Test
    fun `dequant IQ4_NL yields table values`() {
        // d = 1.0, qs all 0x88 -> value kvalues_iq4nl[8] = 1
        val raw = ByteArray(2 + 16) { idx ->
            when (idx) {
                0 -> 0x00; 1 -> 0x3C
                else -> 0x88.toByte()
            }
        }.toList()
        val out = LlamaWeightLoader.dequantIQ4NL(raw, 32)
        assertContentEquals(FloatArray(32) { 1f }.toList(), out.toList())
    }

    @Test
    fun `dequant IQ4_XS applies block scale`() {
        // block0 ls = 33 -> dl = 1, codes 0x88 -> 1; other blocks ls=32 -> dl=0
        val raw = ByteArray(2 + 2 + 4 + 128) { 0x00 }
        raw[0] = 0x00; raw[1] = 0x3C // d = 1.0
        raw[2] = 0xAA.toByte(); raw[3] = 0xAA.toByte() // scales_h = 0xAAAA (ls base 32)
        raw[4] = 0x01 // scales_l first nibble -> ls 33 for block 0
        // qs = 0x88 so value = 1 when dl != 0
        repeat(128) { raw[8 + it] = 0x88.toByte() }
        val out = LlamaWeightLoader.dequantIQ4XS(raw.toList(), 256)
        val expected = FloatArray(256) { idx -> if (idx < 32) 1f else 0f }
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q2_K yields ones for first block`() {
        val raw = ByteArray(4 + 16 + 64) { 0x00 }
        // d = 1.0, dmin = 0.0
        raw[0] = 0x00; raw[1] = 0x3C
        raw[2] = 0x00; raw[3] = 0x00
        // block 0: scale idx 15 (->1.0), min idx 0
        raw[4] = 0xF0.toByte()
        // block 0 codes = 1 (0b01) for all 16 values -> bytes 0x55
        repeat(4) { raw[20 + it] = 0x55.toByte() }
        val out = LlamaWeightLoader.dequantQ2K(raw.toList(), 256)
        val expected = FloatArray(256) { if (it < 16) 1f else 0f }
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q3_K uniform codes`() {
        val raw = ByteArray(2 + 32 + 64 + 12) { 0x00 }
        // d = 2.0
        raw[0] = 0x00; raw[1] = 0x40
        // ql = 3 for all -> 0xFF in qs bytes
        repeat(64) { raw[34 + it] = 0xFF.toByte() }
        // scales all 0x3F -> scale index 63 for every block
        repeat(12) { raw[98 + it] = 0xFF.toByte() }
        val out = LlamaWeightLoader.dequantQ3K(raw.toList(), 256)
        val expected = FloatArray(256) { 6f } // q=3, scale=d*1=2 => 6
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q4_K first block uses scale only`() {
        val raw = ByteArray(4 + 12 + 128) { 0x00 }
        // d = 1.0, dmin = 0.0
        raw[0] = 0x00; raw[1] = 0x3C; raw[2] = 0x00; raw[3] = 0x00
        // block 0: scale idx 63, min idx 0 -> scales bits 0..5 set
        raw[4] = 0x3F
        // block 0 codes = 15 -> bytes 0xFF for first 16 bytes (32 vals)
        repeat(16) { raw[16 + it] = 0xFF.toByte() }
        val out = LlamaWeightLoader.dequantQ4K(raw.toList(), 256)
        val expected = FloatArray(256) { if (it < 32) 15f else 0f }
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q5_K picks high bit`() {
        val raw = ByteArray(4 + 12 + 32 + 128) { 0x00 }
        // d = 1.0, dmin = 0.0
        raw[0] = 0x00; raw[1] = 0x3C; raw[2] = 0x00; raw[3] = 0x00
        // block 0: scale idx 63, min idx 0
        raw[4] = 0x3F
        // qh high bits set for first 32 weights
        repeat(4) { raw[16 + it] = 0xFF.toByte() }
        // qs low nibble zero
        val out = LlamaWeightLoader.dequantQ5K(raw.toList(), 256)
        val expected = FloatArray(256) { if (it < 32) 16f else 0f }
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q6_K combines low and high bits`() {
        val raw = ByteArray(2 + 16 + 128 + 64) { 0x00 }
        // d = 1.0
        raw[0] = 0x00; raw[1] = 0x3C
        // block 0 scale idx 127 -> scale 1.0
        raw[2] = 0x7F
        // ql: two values per byte, both 1 -> 0x11 for first 8 bytes (16 vals)
        repeat(8) { raw[18 + it] = 0x11.toByte() }
        val out = LlamaWeightLoader.dequantQ6K(raw.toList(), 256)
        val expected = FloatArray(256) { if (it < 16) 1f else 0f }
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q8_K scales int8 codes`() {
        val raw = ByteArray(4 + 256 + 32) { 0x00 }
        // d = 1.0f
        raw[0] = 0x00; raw[1] = 0x00; raw[2] = 0x80.toByte(); raw[3] = 0x3F
        raw[4] = 0x01; raw[5] = 0x02; raw[6] = 0x03; raw[7] = 0x04
        val out = LlamaWeightLoader.dequantQ8K(raw.toList(), 256)
        val expected = FloatArray(256) { idx ->
            when (idx) {
                0 -> 1f
                1 -> 2f
                2 -> 3f
                3 -> 4f
                else -> 0f
            }
        }
        assertContentEquals(expected.toList(), out.toList())
    }
}
