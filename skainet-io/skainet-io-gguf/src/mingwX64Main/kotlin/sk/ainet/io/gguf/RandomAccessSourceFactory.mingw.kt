package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.WindowsRandomAccessSource

/**
 * Windows implementation of [createRandomAccessSource]: `ReadFile` with an `OVERLAPPED`
 * offset via io-core's [WindowsRandomAccessSource] (posix `pread` does not exist on
 * mingw). Returns `null` if the file cannot be opened, matching the JVM/POSIX actuals'
 * contract so callers can fall back to the legacy sequential reader. See #911.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? =
    WindowsRandomAccessSource.open(filePath)
