package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.openRandomAccessSource

/**
 * Open a GGUF file for positional reads, or `null` where the platform has no file system.
 *
 * @deprecated One declaration for every format now lives in `skainet-io-core` (#1037): this module,
 *   `-safetensors` and `-onnx` each carried an identical `expect fun` with six platform actuals,
 *   and they had already drifted apart on Kotlin/Native.
 */
@Deprecated(
    message = "The per-format source factories are one function in skainet-io-core (SKEEP-003 §7, #1037).",
    replaceWith = ReplaceWith("openRandomAccessSource(filePath)", "sk.ainet.io.openRandomAccessSource"),
)
public fun createRandomAccessSource(filePath: String): RandomAccessSource? = openRandomAccessSource(filePath)
