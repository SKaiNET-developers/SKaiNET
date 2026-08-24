package sk.ainet.io

/**
 * androidNativeArm32: no positional source. `PosixPreadRandomAccessSource` lives in `native64Main`
 * because 32-bit `ssize_t`/`size_t` are `Int` here and `Long` everywhere else (see the target
 * comment in this module's build script); on-device file I/O for arm32 is its own concern.
 */
public actual fun openRandomAccessSource(filePath: String): RandomAccessSource? = null
