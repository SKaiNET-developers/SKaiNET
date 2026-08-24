package sk.ainet.io

/** No file system to open a path against — a browser or Wasm host reads a model through a Blob or a fetch. */
public actual fun openRandomAccessSource(filePath: String): RandomAccessSource? = null
