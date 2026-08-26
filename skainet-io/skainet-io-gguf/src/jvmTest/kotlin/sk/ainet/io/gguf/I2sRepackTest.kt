package sk.ainet.io.gguf

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1140: the group→sequential repack against the packing rule of BitNet.cpp's `quantize_i2_s`
 * (element `j` of a `QK`-element block → byte `j % (QK/4)`, bit-pair `6 − 2·(j / (QK/4))`), for
 * both of its architecture-dependent flavors, plus the fail-fast on byte code 3.
 */
class I2sRepackTest {

    /** The BitNet.cpp packing, reimplemented independently as the test oracle. */
    private fun packGroup(codes: IntArray, qk: Int): ByteArray {
        val bytesPerBlock = qk / 4
        val out = ByteArray(codes.size / 4)
        for (j in codes.indices) {
            val jb = j % qk
            val byteIndex = (j / qk) * bytesPerBlock + jb % bytesPerBlock
            out[byteIndex] = (out[byteIndex].toInt() or (codes[j] shl (6 - 2 * (jb / bytesPerBlock)))).toByte()
        }
        return out
    }

    private fun packSequential(codes: IntArray): ByteArray {
        val out = ByteArray(codes.size / 4)
        for (j in codes.indices) {
            out[j / 4] = (out[j / 4].toInt() or (codes[j] shl ((j % 4) * 2))).toByte()
        }
        return out
    }

    private fun randomCodes(count: Int, seed: Int): IntArray {
        val rng = Random(seed)
        return IntArray(count) { rng.nextInt(3) } // {0, 1, 2} — never 3
    }

    @Test
    fun group128RepacksToTheSequentialOrder() {
        val codes = randomCodes(256, seed = 1)
        val repacked = I2sRepack.toSequentialPayload(packGroup(codes, qk = 128), 256, I2sGgufLayout.GROUP_128)
        assertContentEquals(packSequential(codes), repacked)
    }

    @Test
    fun group64RepacksToTheSequentialOrder() {
        val codes = randomCodes(256, seed = 2)
        val repacked = I2sRepack.toSequentialPayload(packGroup(codes, qk = 64), 256, I2sGgufLayout.GROUP_64)
        assertContentEquals(packSequential(codes), repacked)
    }

    @Test
    fun sequentialPassesThroughValidatedAndTruncatedToThePayload() {
        val codes = randomCodes(64, seed = 3)
        val withTrailer = packSequential(codes) + ByteArray(32) { 0x7F } // trailer must be ignored
        val out = I2sRepack.toSequentialPayload(withTrailer, 64, I2sGgufLayout.SEQUENTIAL)
        assertContentEquals(packSequential(codes), out)
    }

    @Test
    fun byteCode3FailsFastInEveryLayout() {
        for (layout in I2sGgufLayout.entries) {
            val bytes = ByteArray(128 / 4) // 128 elements of code 0...
            bytes[0] = 0xC0.toByte() // ...except one bit-pair of 3
            val e = assertFailsWith<IllegalArgumentException>("layout $layout") {
                I2sRepack.toSequentialPayload(bytes, 128, layout)
            }
            assertTrue("code 3" in e.message!!, e.message!!)
        }
    }

    @Test
    fun wrongBlockMultipleNamesTheOtherFlavor() {
        // 64 elements fill a GROUP_64 block but not a GROUP_128 one.
        val e = assertFailsWith<IllegalArgumentException> {
            I2sRepack.toSequentialPayload(ByteArray(16), 64, I2sGgufLayout.GROUP_128)
        }
        assertTrue("GROUP_64" in e.message!!, e.message!!)
    }

    @Test
    fun scaleTrailerIsLittleEndianFp32() {
        val buffer = I2sRepack.withScale(ByteArray(4), 0.25f)
        val bits = (buffer[4].toInt() and 0xFF) or
            ((buffer[5].toInt() and 0xFF) shl 8) or
            ((buffer[6].toInt() and 0xFF) shl 16) or
            ((buffer[7].toInt() and 0xFF) shl 24)
        assertContentEquals(listOf(8), listOf(buffer.size))
        assertTrue(Float.fromBits(bits) == 0.25f)
    }
}
