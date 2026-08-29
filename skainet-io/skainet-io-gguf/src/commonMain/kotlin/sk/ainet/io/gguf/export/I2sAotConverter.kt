package sk.ainet.io.gguf.export

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.I2sGgufLayout
import sk.ainet.io.gguf.I2sRepack
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingTensorInfo
import sk.ainet.io.gguf.i2sCompanionScaleTensor
import sk.ainet.io.gguf.resolveI2sScale

/**
 * AOT conversion (#1207): reads an arbitrary GGUF file and re-encodes any I2_S (ternary/BitNet)
 * tensors into a `SEQUENTIAL`, trailer-scaled buffer — the exact shape #1203's loader serves
 * zero-copy from mmap, with no runtime repack at all. Every other tensor, and every file-level
 * KV metadata entry, passes through byte-for-byte/value-for-value unchanged.
 *
 * This is the "convert once, ahead of time" alternative to caching the repack on-device
 * (#1204): a build that controls its own model pipeline runs this once, ships the converted
 * file, and every future load of it is a genuine zero-copy mmap — no sidecar, no first-load
 * penalty on the constrained device the conversion exists to protect. See #1198 for the full
 * reasoning behind preferring this over the on-device cache.
 *
 * Scope: this converts the file *content* a consuming loader reads (tensors + KV metadata). It
 * does not attempt to reconstruct duplicate top-level KV keys — a pathological, practically
 * unseen case `StreamingGGUFReader` itself only handles defensively (renaming them
 * `<key>_dup_N`) — those would round-trip as literal `_dup_N`-suffixed keys rather than true
 * duplicates.
 */
public object I2sAotConverter {

    /** Reader-synthetic keys ([StreamingGGUFReader.fields] adds these; they are not real GGUF KV entries — [GGUFWriter] computes its own header counts/version). */
    private val SYNTHETIC_KEYS = setOf("GGUF.tensor_count", "GGUF.kv_count", "GGUF.version")

    /**
     * Build the [GgufWriteRequest] for converting [source]'s tensors, whose I2_S tensors (if
     * any) are in [sourceLayout] order (see [I2sGgufLayout] — a property of the converter that
     * wrote the *source* file, not recoverable from its bytes). The caller writes the result
     * with [GGUFWriter], e.g. `GGUFWriter.writeToSink(convert(source, layout), sink)`.
     */
    public fun convert(source: RandomAccessSource, sourceLayout: I2sGgufLayout): GgufWriteRequest {
        StreamingGGUFReader.open(source).use { reader ->
            val tensors = reader.tensors
            val entries = ArrayList<GgufTensorEntry>(tensors.size)
            for (info in tensors) {
                when {
                    info.tensorType == GGMLQuantizationType.I2_S -> {
                        val scale = resolveI2sScale(info, tensors, reader, source, sourceLayout)
                        val sequential = I2sRepack.toSequentialPayload(
                            reader.loadTensorData(info),
                            info.nElements.toInt(),
                            sourceLayout,
                        )
                        entries += GgufTensorEntry(
                            ggufName = info.name,
                            quantization = GGMLQuantizationType.I2_S,
                            shape = info.shape.map { it.toInt() },
                            rawBytes = I2sRepack.withScale(sequential, scale),
                        )
                    }
                    // Its value is now folded into the trailer above — the converted I2_S
                    // tensor is self-contained, so the companion scalar is no longer needed
                    // (and keeping it would defeat #1203's mmap fast path, which requires no
                    // companion tensor to exist at all).
                    isI2sCompanionScale(info, tensors) -> Unit
                    else -> entries += GgufTensorEntry(
                        ggufName = info.name,
                        quantization = info.tensorType,
                        shape = info.shape.map { it.toInt() },
                        rawBytes = reader.loadTensorData(info),
                    )
                }
            }
            val metadata = LinkedHashMap<String, Any>(reader.fields.size)
            for ((key, value) in reader.fields) {
                if (key in SYNTHETIC_KEYS || value == null) continue
                metadata[key] = value
            }
            return GgufWriteRequest(metadata = metadata, tensors = entries, tensorMap = emptyMap())
        }
    }

    private fun isI2sCompanionScale(tensorInfo: StreamingTensorInfo, tensors: List<StreamingTensorInfo>): Boolean =
        tensorInfo.tensorType == GGMLQuantizationType.F32 &&
            tensors.any { it.tensorType == GGMLQuantizationType.I2_S && i2sCompanionScaleTensor(it, tensors) == tensorInfo }
}
