package sk.ainet.io.irpa

import kotlinx.io.Sink
import sk.ainet.compile.hlo.ConstantSource

/**
 * Stream the bytes of a [ConstantSource.FileRange] directly into [sink].
 *
 * The whole point of a file range is that the tensor lives
 * as a byte range in a source file — the GGUF or safetensors blob on
 * disk. Under [IrpaWriter], that range blits verbatim into the
 * `.irpa` archive's storage segment with no intermediate heap copy,
 * no parse, and no re-quantization. PR E of issue #523 closes the
 * loop on this path for real models where inline weights are
 * unworkable (Whisper-tiny ≈ 151 MB text MLIR under the inline
 * policy).
 *
 * JVM actual uses `FileChannel.map` for a true mmap window. Platforms
 * without mmap support throw with a pointer to the tracking issue —
 * rather than silently falling back to a slower read path that would
 * undermine the "zero-copy ingestion" contract callers rely on.
 *
 * Implementations MUST:
 *  - Respect [ConstantSource.FileRange.fileOffset] as the starting
 *    byte in the file, not in any mapped window.
 *  - Write exactly [ConstantSource.FileRange.sizeInBytes] bytes.
 *  - Not flush or close [sink]; the caller manages lifecycle.
 *  - Close any OS resources they open (file descriptors, mapped
 *    regions) before returning, regardless of exceptions.
 */
internal expect fun writeFileRangeBytes(sink: Sink, handle: ConstantSource.FileRange)
