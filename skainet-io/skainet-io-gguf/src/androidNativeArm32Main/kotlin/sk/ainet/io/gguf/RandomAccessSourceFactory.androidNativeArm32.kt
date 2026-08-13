package sk.ainet.io.gguf

import sk.ainet.io.Posix32PreadRandomAccessSource
import sk.ainet.io.RandomAccessSource

/**
 * `androidNativeArm32` implementation of [createRandomAccessSource] using POSIX
 * `pread(2)` via [Posix32PreadRandomAccessSource] — the 32-bit-`off_t` sibling of
 * [sk.ainet.io.PosixPreadRandomAccessSource] (native64Main), kept as a separate
 * actual because this target can't share a source set with the 64-bit natives
 * (mixed `ssize_t`/`off_t` widths fail Kotlin/Native's metadata compile).
 *
 * Returns `null` if the file cannot be opened, matching every other actual's
 * contract so callers fall back to the legacy sequential reader.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? =
    Posix32PreadRandomAccessSource.open(filePath)
