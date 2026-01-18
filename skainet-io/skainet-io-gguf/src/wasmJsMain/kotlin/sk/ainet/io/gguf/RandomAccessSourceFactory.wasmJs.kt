package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource

/**
 * WasmJS implementation of [createRandomAccessSource].
 *
 * Returns null as WasmJS doesn't have efficient random file access.
 * Callers should fall back to legacy GGUFReader which loads the full file.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? = null
