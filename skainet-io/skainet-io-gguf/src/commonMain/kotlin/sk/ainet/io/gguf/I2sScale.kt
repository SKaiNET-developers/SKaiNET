package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource

/**
 * I2_S scale resolution (#1140), shared by [StreamingGgufParametersLoader] and any other reader
 * of an I2_S GGUF — notably an AOT converter (#1207) that needs the exact same answer the
 * streaming loader would give, without duplicating the decision.
 */

/** The `<name>_scale` companion tensor for an I2_S weight, if the converter wrote one (#1140). */
internal fun i2sCompanionScaleTensor(
    tensorInfo: StreamingTensorInfo,
    tensors: List<StreamingTensorInfo>,
): StreamingTensorInfo? = tensors.firstOrNull {
    it.tensorType == GGMLQuantizationType.F32 && it.name == "${tensorInfo.name}_scale"
}

private fun le32(bytes: ByteArray, offset: Int = 0): Float = Float.fromBits(
    (bytes[offset].toInt() and 0xFF) or
        ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
        ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
        ((bytes[offset + 3].toInt() and 0xFF) shl 24)
)

/**
 * The little-endian FP32 immediately after [tensorInfo]'s payload — BitNet.cpp's trailer
 * convention — or `null` if unreadable, non-finite, or zero. [StreamingTensorInfo.nBytes]
 * deliberately sizes the payload only, so `absoluteDataOffset + nBytes` is exactly where a
 * trailer would start.
 */
internal fun i2sTrailerScale(tensorInfo: StreamingTensorInfo, source: RandomAccessSource): Float? =
    runCatching { le32(source.readAt(tensorInfo.absoluteDataOffset + tensorInfo.nBytes, 4)) }
        .getOrNull()?.takeIf { it.isFinite() && it != 0f }

/**
 * The per-tensor FP32 scale of an I2_S weight, from wherever its converter put it (#1140):
 *
 * - **BitNet.cpp** writes it as a trailer after the payload (a 32-byte-aligned region whose
 *   first 4 bytes are the LE FP32 scale; `w = (code − 1) · scale`). Read directly from the
 *   source at `absoluteDataOffset + payload` — [StreamingTensorInfo.nBytes] deliberately sizes
 *   the payload only.
 * - **NeoGPU's converter** writes a companion `<name>_scale` F32 scalar, defined as "divide the
 *   projection output by it" — so the stored multiplier is its inverse.
 * - Neither present (or unreadable/non-finite/zero): `1.0`, i.e. the raw codes. Loud in the
 *   trace via the repack conversion's byte counts, never a crash.
 *
 * [layout] decides which source is tried first; both are accepted either way, because a
 * sequential file with a trailer or a group file with a companion costs nothing to honour.
 */
internal fun resolveI2sScale(
    tensorInfo: StreamingTensorInfo,
    tensors: List<StreamingTensorInfo>,
    reader: StreamingGGUFReader,
    source: RandomAccessSource,
    layout: I2sGgufLayout,
): Float {
    fun companionInverse(): Float? {
        val companion = i2sCompanionScaleTensor(tensorInfo, tensors) ?: return null
        val value = runCatching { le32(reader.loadTensorData(companion)) }.getOrNull() ?: return null
        if (!value.isFinite() || value == 0f) return null
        return 1f / value
    }

    return when (layout) {
        I2sGgufLayout.GROUP_128, I2sGgufLayout.GROUP_64 -> i2sTrailerScale(tensorInfo, source) ?: companionInverse() ?: 1f
        I2sGgufLayout.SEQUENTIAL -> companionInverse() ?: i2sTrailerScale(tensorInfo, source) ?: 1f
    }
}

/**
 * Whether an I2_S tensor's on-disk bytes are, as-is, a complete kernel-ready `BITNET_B1_58`
 * buffer — payload immediately followed by its own trailing FP32 scale — so mapping
 * `[absoluteDataOffset, absoluteDataOffset + nBytes + 4)` directly gives exactly what
 * [resolveI2sScale] would have computed anyway (#1203).
 *
 * Only ever true for [I2sGgufLayout.SEQUENTIAL]: the payload itself is already in
 * `BITNET_B1_58` order there, whereas `GROUP_128`/`GROUP_64` payloads still need permuting
 * regardless of where the scale lives. False whenever a companion `<name>_scale` tensor exists
 * — [resolveI2sScale]'s `SEQUENTIAL` order prefers it over a trailer — or the trailer bytes
 * don't parse to a finite, nonzero float.
 */
internal fun i2sTrailerScaleIsMappable(
    tensorInfo: StreamingTensorInfo,
    tensors: List<StreamingTensorInfo>,
    source: RandomAccessSource,
    layout: I2sGgufLayout,
): Boolean =
    layout == I2sGgufLayout.SEQUENTIAL &&
        i2sCompanionScaleTensor(tensorInfo, tensors) == null &&
        i2sTrailerScale(tensorInfo, source) != null
