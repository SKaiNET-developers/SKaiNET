package sk.ainet.io.onnx

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.openRandomAccessSource

/** Open an ONNX file for positional reads, or `null` where the platform has no file system. */
@Deprecated(
    message = "The per-format source factories are one function in skainet-io-core (SKEEP-003 §7, #1037).",
    replaceWith = ReplaceWith("openRandomAccessSource(filePath)", "sk.ainet.io.openRandomAccessSource"),
)
public fun createOnnxRandomAccessSource(filePath: String): RandomAccessSource? = openRandomAccessSource(filePath)
