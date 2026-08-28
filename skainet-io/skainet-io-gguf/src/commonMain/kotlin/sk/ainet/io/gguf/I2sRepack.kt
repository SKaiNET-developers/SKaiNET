package sk.ainet.io.gguf

/*
 * The I2_S layout rules in this file are interpreted from the sources of
 * "https://github.com/microsoft/BitNet" (BitNet.cpp, MIT, © Microsoft Corporation) —
 * `quantize_i2_s` in src/ggml-bitnet-mad.cpp — and from NeoGPU's
 * tools/convert_bitnet_to_gguf.py ("https://github.com/anjaustin/neogpu", MIT).
 * A clean-room reimplementation of the format, not copied code — the same
 * convention as the llama.cpp note in Constants.kt.
 *
 * SPDX-FileCopyrightText:  Copyright (c) Microsoft Corporation
 * SPDX-License-Identifier: MIT
 */
/**
 * Which bit order an I2_S GGUF's payload is in — a property of the *converter that wrote the
 * file*, not recoverable from the bytes, so the caller has to say (#1140).
 *
 * BitNet.cpp's `quantize_i2_s` packs 2-bit codes into fixed-size blocks, element `j` of a block
 * going to byte `j % (QK/4)`, bit-pair `6 − 2·(j / (QK/4))` (high bits first) — and `QK_I2_S`
 * is **128 when the file was quantized on x86 (AVX) and 64 on ARM (NEON)**, an architecture-
 * dependent file format. NeoGPU's `convert_bitnet_to_gguf.py` packs the same codes sequentially,
 * four consecutive elements per byte, low bit-pair first — byte-identical to SKaiNET's
 * `BITNET_B1_58` payload. All three agree on the code mapping `{0,1,2} → {-1,0,+1}`.
 */
public enum class I2sGgufLayout(internal val blockElements: Int) {
    /** BitNet.cpp file quantized with the x86/AVX pipeline (`QK_I2_S = 128`, 32-byte blocks). The common case for published GGUFs. */
    GROUP_128(128),

    /** BitNet.cpp file quantized with the ARM/NEON pipeline (`QK_I2_S = 64`, 16-byte blocks). */
    GROUP_64(64),

    /** NeoGPU's converter: sequential 4-per-byte, low bit-pair first — already the `BITNET_B1_58` payload order. */
    SEQUENTIAL(4),
}

/**
 * Repacks an I2_S payload into the sequential `BITNET_B1_58` payload order (#1140).
 *
 * Byte code 3 is **rejected here, at import** — it has no ternary meaning (the kernels decode it
 * as +2 by LUT arithmetic, deliberately unvalidated), so a file that contains it is corrupt and
 * the load fails fast instead of silently producing garbage logits.
 */
public object I2sRepack {

    /**
     * The sequential `BITNET_B1_58` payload of [elementCount] codes read from [bytes] under
     * [layout]. For [I2sGgufLayout.SEQUENTIAL] the payload is validated and, when [bytes] holds
     * exactly [elementCount]'s payload with no trailing slack to trim, returned as the same
     * array reference — no copy (#1203): `SEQUENTIAL` is already the target byte order, so the
     * only thing standing between it and a zero-copy load was this method defensively trimming a
     * buffer that didn't need trimming.
     *
     * @throws IllegalArgumentException on byte code 3, or when [elementCount] does not fill
     *   [layout]'s blocks exactly
     */
    public fun toSequentialPayload(bytes: ByteArray, elementCount: Int, layout: I2sGgufLayout): ByteArray {
        require(elementCount % 4 == 0) { "I2_S element count must be a multiple of 4; got $elementCount" }
        val payloadBytes = elementCount / 4
        require(bytes.size >= payloadBytes) {
            "I2_S payload needs $payloadBytes bytes for $elementCount elements; got ${bytes.size}"
        }
        if (layout == I2sGgufLayout.SEQUENTIAL) {
            for (i in 0 until payloadBytes) {
                val b = bytes[i].toInt() and 0xFF
                for (lane in 0 until 4) {
                    val element = i * 4 + lane
                    if (element >= elementCount) break
                    requireValidCode((b shr (lane * 2)) and 3, element)
                }
            }
            return if (bytes.size == payloadBytes) bytes else bytes.copyOf(payloadBytes)
        }
        val qk = layout.blockElements
        require(elementCount % qk == 0) {
            "a ${layout.name} I2_S tensor must be a multiple of $qk elements; got $elementCount " +
                "(the file may be the other BitNet.cpp flavor — try ${otherGroup(layout).name})"
        }
        val bytesPerBlock = qk / 4
        val out = ByteArray(payloadBytes)
        for (element in 0 until elementCount) {
            val jb = element % qk
            val src = bytes[(element / qk) * bytesPerBlock + jb % bytesPerBlock].toInt() and 0xFF
            val code = (src shr (6 - 2 * (jb / bytesPerBlock))) and 3
            requireValidCode(code, element)
            val shift = (element % 4) * 2
            out[element / 4] = (out[element / 4].toInt() or (code shl shift)).toByte()
        }
        return out
    }

    /** [payload] with [scale] appended as the little-endian FP32 trailer — a complete `BITNET_B1_58` buffer. */
    public fun withScale(payload: ByteArray, scale: Float): ByteArray {
        val out = payload.copyOf(payload.size + 4)
        val bits = scale.toRawBits()
        out[payload.size] = (bits and 0xFF).toByte()
        out[payload.size + 1] = ((bits shr 8) and 0xFF).toByte()
        out[payload.size + 2] = ((bits shr 16) and 0xFF).toByte()
        out[payload.size + 3] = ((bits shr 24) and 0xFF).toByte()
        return out
    }

    private fun requireValidCode(code: Int, element: Int) {
        require(code != 3) {
            "I2_S element $element holds byte code 3, which is not a ternary value — the file is " +
                "corrupt, or it was read under the wrong I2sGgufLayout"
        }
    }

    private fun otherGroup(layout: I2sGgufLayout): I2sGgufLayout =
        if (layout == I2sGgufLayout.GROUP_128) I2sGgufLayout.GROUP_64 else I2sGgufLayout.GROUP_128
}
