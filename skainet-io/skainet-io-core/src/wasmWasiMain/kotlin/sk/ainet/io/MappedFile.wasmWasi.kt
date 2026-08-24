package sk.ainet.io

/** No file mapping here — staging falls back to the heap. */
public actual fun openMappedFile(filePath: String): MappedFile? = null
