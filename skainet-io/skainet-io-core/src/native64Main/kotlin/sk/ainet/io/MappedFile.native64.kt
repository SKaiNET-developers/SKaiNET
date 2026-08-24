package sk.ainet.io

/**
 * Kotlin/Native: not yet. `mmap(2)` plus a `TensorData` over a `CPointer` is the natural
 * implementation and belongs with the native `Storage.Mapped` work (#1020); until then native
 * staging falls back to the heap rather than pretending.
 */
public actual fun openMappedFile(filePath: String): MappedFile? = null
