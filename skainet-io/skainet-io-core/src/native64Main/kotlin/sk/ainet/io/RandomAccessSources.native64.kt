package sk.ainet.io

/** 64-bit Kotlin/Native: POSIX `pread(2)` ([PosixPreadRandomAccessSource]); `null` if it cannot open. */
public actual fun openRandomAccessSource(filePath: String): RandomAccessSource? =
    PosixPreadRandomAccessSource.open(filePath)
