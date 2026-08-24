package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.openRandomAccessSource

/**
 * Open a safetensors file for positional reads, or `null` where the platform has no file system.
 *
 * Note this now streams on Kotlin/Native too: the copy this delegates to uses `pread(2)`, while
 * this module's own native actual used to return `null` — the drift #1037 removed.
 */
@Deprecated(
    message = "The per-format source factories are one function in skainet-io-core (SKEEP-003 §7, #1037).",
    replaceWith = ReplaceWith("openRandomAccessSource(filePath)", "sk.ainet.io.openRandomAccessSource"),
)
public fun createRandomAccessSource(filePath: String): RandomAccessSource? = openRandomAccessSource(filePath)
