package sk.ainet.models.llama

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
    fun `dequant Q8_1 scales codes correctly`() {
        // Q8_1 uses f32 for d and s (not f16)
        // d=1.0f (0x3F800000 little-endian: 0x00 0x00 0x80 0x3F)
        // s=0.0f (not used in dequant, just stored)
        // qs = 1, 2, 3, 4, ... for first 4 values, rest zeros
        val raw = ByteArray(8 + 32) { 0x00 }
        // d = 1.0f in little-endian
        raw[0] = 0x00; raw[1] = 0x00; raw[2] = 0x80.toByte(); raw[3] = 0x3F
        // s = 0.0f (unused)
        raw[4] = 0x00; raw[5] = 0x00; raw[6] = 0x00; raw[7] = 0x00
        // qs[0..3] = 1, 2, 3, 4
        raw[8] = 0x01; raw[9] = 0x02; raw[10] = 0x03; raw[11] = 0x04
        val out = LlamaWeightLoader.dequantQ8_1(raw.toList(), 32)
        val expected = FloatArray(32) { idx ->
            when (idx) {
                0 -> 1f; 1 -> 2f; 2 -> 3f; 3 -> 4f
                else -> 0f
            }
        }
        assertContentEquals(expected.toList(), out.toList())
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
        val raw = ByteArray(144) { 0x00 }
        // d = 1.0, dmin = 0.0
        raw[0] = 0x00; raw[1] = 0x3C; raw[2] = 0x00; raw[3] = 0x00
        // block 0: sc=1, m=0 (from getScaleMinK4 logic)
        // Group 0 uses scales[0] for sc1, scales[1] for sc2
        raw[4] = 0x01
        raw[5] = 0x01
        // block 0 codes = 15 -> bytes 0xFF for first 32 bytes of qs (64 vals)
        repeat(32) { raw[16 + it] = 0xFF.toByte() }
        val out = LlamaWeightLoader.dequantQ4K(raw.toList(), 256)
        val expected = FloatArray(256) { if (it < 64) 15f else 0f }
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q5_K picks high bit`() {
        val raw = ByteArray(176) { 0x00 }
        // d = 1.0, dmin = 0.0
        raw[0] = 0x00; raw[1] = 0x3C; raw[2] = 0x00; raw[3] = 0x00
        // block 0: sc=1, m=0
        raw[4] = 0x01
        raw[5] = 0x01
        // qh high bits set for first 32 weights
        repeat(4) { raw[16 + it] = 0xFF.toByte() }
        // qs low nibbles zero
        val out = LlamaWeightLoader.dequantQ5K(raw.toList(), 256)
        val expected = FloatArray(256) { if (it < 32) 16f else 0f }
        assertContentEquals(expected.toList(), out.toList())
    }

    @Test
    fun `dequant Q6_K combines low and high bits`() {
        val raw = ByteArray(210) { 0x00 }
        // d = 1.0
        raw[208] = 0x00; raw[209] = 0x3C
        // scales at offset 192
        // Both halves (128 elements each) should have scale=1 for all groups
        repeat(16) { raw[192 + it] = 0x01 }

        // ql: offset 0. 0x21 -> becomes (1|2<<4)-32 = 33-32 = 1.
        repeat(128) { raw[it] = 0x21.toByte() }
        // qh[0] bits 0-1 = 0 (00), 2-3 = 0, 4-5 = 0, 6-7 = 0 -> 0x00
        // wait, if ql = 0x21, qLow=1, qHigh=2.
        // Actually, if ql=0x21, then:
        // q1Low = ql[l] & 0x0F = 1
        // q3Low = ql[l] >> 4 = 2
        // q2Low = ql[l+32] & 0x0F = 1
        // q4Low = ql[l+32] >> 4 = 2
        // We need (qLow | qHigh << 4) = 33.
        // For q1: q1Low=1, so q1High=2. qh bits 0-1 = 2.
        // For q2: q2Low=1, so q2High=2. qh bits 2-3 = 2.
        // For q3: q3Low=2, so q3High=? (2 | q3High<<4) = 33 -> 31/16 not integer.
        // Let's use simpler: qLow=1, qHigh=2 -> 33.
        // q1Low=1, q1High=2.
        // q2Low=1, q2High=2.
        // q3Low=1, q3High=2.
        // q4Low=1, q4High=2.
        // So ql all 0x11, qh all 0xAA.
        repeat(128) { raw[it] = 0x11.toByte() }
        repeat(64) { raw[128 + it] = 0xAA.toByte() }

        val out = LlamaWeightLoader.dequantQ6K(raw.toList(), 256)
        val expected = FloatArray(256) { 1f }
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

    @Test
    fun `dequant TQ2_0 block with scale 1 and all zeros yields minus ones`() {
        // TQ2_0: 66 bytes = 64 data + 2 f16 scale
        // All data bytes = 0x00 -> each 2-bit value is 0 -> (0-1) = -1
        // Scale = 1.0 (0x3C00)
        val raw = ByteArray(66) { 0x00 }
        raw[64] = 0x00  // scale low byte
        raw[65] = 0x3C  // scale high byte (f16 1.0)
        val out = LlamaWeightLoader.dequantTQ2_0(raw.toList(), 256)
        assertContentEquals(FloatArray(256) { -1f }.toList(), out.toList())
    }

    @Test
    fun `dequant TQ2_0 block with all ones yields zeros`() {
        // All data bytes = 0x55 -> each 2-bit value is 1 (01 01 01 01) -> (1-1) = 0
        // Scale = 1.0
        val raw = ByteArray(66) { 0x55 }
        raw[64] = 0x00; raw[65] = 0x3C
        val out = LlamaWeightLoader.dequantTQ2_0(raw.toList(), 256)
        assertContentEquals(FloatArray(256) { 0f }.toList(), out.toList())
    }

    @Test
    fun `dequant TQ2_0 block with all twos yields plus ones`() {
        // All data bytes = 0xAA -> each 2-bit value is 2 (10 10 10 10) -> (2-1) = +1
        // Scale = 1.0
        val raw = ByteArray(66) { 0xAA.toByte() }
        raw[64] = 0x00; raw[65] = 0x3C
        val out = LlamaWeightLoader.dequantTQ2_0(raw.toList(), 256)
        assertContentEquals(FloatArray(256) { 1f }.toList(), out.toList())
    }

    @Test
    fun `dequant TQ2_0 block applies scale correctly`() {
        // All twos (+1) with scale = 2.0 (0x4000)
        val raw = ByteArray(66) { 0xAA.toByte() }
        raw[64] = 0x00; raw[65] = 0x40  // f16 2.0
        val out = LlamaWeightLoader.dequantTQ2_0(raw.toList(), 256)
        assertContentEquals(FloatArray(256) { 2f }.toList(), out.toList())
    }

    @Test
    fun `dequant TQ2_0 block with mixed values`() {
        // First byte = 0xE4 = 11 10 01 00 in binary
        // Values: v0=0 (-1), v1=1 (0), v2=2 (+1), v3=3 -> but 3 is invalid, should be clamped to +2
        // Actually TQ2_0 only uses values 0,1,2. If we see 3, (3-1)=2
        val raw = ByteArray(66) { 0x55 }  // default to zeros
        raw[0] = 0xE4.toByte()  // 11_10_01_00: v0=-1, v1=0, v2=+1, v3=+2 (if 3 is allowed)
        raw[64] = 0x00; raw[65] = 0x3C  // scale = 1.0
        val out = LlamaWeightLoader.dequantTQ2_0(raw.toList(), 256)
        // First 4 elements: (0-1)=-1, (1-1)=0, (2-1)=+1, (3-1)=+2
        kotlin.test.assertEquals(-1f, out[0], 0.001f)
        kotlin.test.assertEquals(0f, out[1], 0.001f)
        kotlin.test.assertEquals(1f, out[2], 0.001f)
        kotlin.test.assertEquals(2f, out[3], 0.001f)  // 3 encodes as +2 when scaled
    }

    @Test
    fun `dequant TQ1_0 block with all zeros yields minus ones`() {
        // TQ1_0: 54 bytes = 48 base-3 + 4 2-bit + 2 f16 scale
        // All base-3 bytes = 0 means each decoded value is 0 -> (0-1) = -1
        // All 2-bit bytes = 0 means remaining 16 values are also -1
        val raw = ByteArray(54) { 0x00 }
        raw[52] = 0x00; raw[53] = 0x3C  // scale = 1.0
        val out = LlamaWeightLoader.dequantTQ1_0(raw.toList(), 256)
        assertContentEquals(FloatArray(256) { -1f }.toList(), out.toList())
    }

    @Test
    fun `dequant TQ1_0 block with base3 ones yields zeros`() {
        // Base-3 encoding: each byte encodes 5 values as v0 + v1*3 + v2*9 + v3*27 + v4*81
        // For all ones: 1 + 3 + 9 + 27 + 81 = 121 (0x79)
        // 2-bit packed: 0x55 = 01 01 01 01 = all ones
        val raw = ByteArray(54) { 0x00 }
        repeat(48) { raw[it] = 0x79 }  // base-3 all ones
        repeat(4) { raw[48 + it] = 0x55 }  // 2-bit all ones
        raw[52] = 0x00; raw[53] = 0x3C  // scale = 1.0
        val out = LlamaWeightLoader.dequantTQ1_0(raw.toList(), 256)
        assertContentEquals(FloatArray(256) { 0f }.toList(), out.toList())
    }

    @Test
    fun `dequant TQ1_0 block with base3 twos yields plus ones`() {
        // For all twos: 2 + 6 + 18 + 54 + 162 = 242 (0xF2)
        // 2-bit packed: 0xAA = 10 10 10 10 = all twos
        val raw = ByteArray(54) { 0x00 }
        repeat(48) { raw[it] = 0xF2.toByte() }  // base-3 all twos
        repeat(4) { raw[48 + it] = 0xAA.toByte() }  // 2-bit all twos
        raw[52] = 0x00; raw[53] = 0x3C  // scale = 1.0
        val out = LlamaWeightLoader.dequantTQ1_0(raw.toList(), 256)
        assertContentEquals(FloatArray(256) { 1f }.toList(), out.toList())
    }

    @Test
    fun `dequant TQ1_0 block applies scale correctly`() {
        // All twos with scale = 2.0
        val raw = ByteArray(54) { 0x00 }
        repeat(48) { raw[it] = 0xF2.toByte() }  // base-3 all twos
        repeat(4) { raw[48 + it] = 0xAA.toByte() }  // 2-bit all twos
        raw[52] = 0x00; raw[53] = 0x40  // scale = 2.0
        val out = LlamaWeightLoader.dequantTQ1_0(raw.toList(), 256)
        assertContentEquals(FloatArray(256) { 2f }.toList(), out.toList())
    }

    @Test
    fun `dequant TQ1_0 base3 decoding for mixed values`() {
        // Test decoding first 5 values from one base-3 byte
        // Values: 0, 1, 2, 0, 1 -> 0 + 1*3 + 2*9 + 0*27 + 1*81 = 3 + 18 + 81 = 102 (0x66)
        val raw = ByteArray(54) { 0x79 }  // default all ones
        raw[0] = 0x66  // first 5 values: -1, 0, +1, -1, 0
        repeat(4) { raw[48 + it] = 0x55 }  // 2-bit all ones
        raw[52] = 0x00; raw[53] = 0x3C  // scale = 1.0
        val out = LlamaWeightLoader.dequantTQ1_0(raw.toList(), 256)
        kotlin.test.assertEquals(-1f, out[0], 0.001f)
        kotlin.test.assertEquals(0f, out[1], 0.001f)
        kotlin.test.assertEquals(1f, out[2], 0.001f)
        kotlin.test.assertEquals(-1f, out[3], 0.001f)
        kotlin.test.assertEquals(0f, out[4], 0.001f)
    }
}
