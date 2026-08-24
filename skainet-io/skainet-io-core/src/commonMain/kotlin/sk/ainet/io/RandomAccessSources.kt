package sk.ainet.io

/**
 * Open [filePath] for positional reads, or return `null` when this platform has no file system to
 * read it from (JS, Wasm) or the file cannot be opened.
 *
 * One declaration for the whole project (#1037): `skainet-io-gguf`, `-safetensors` and `-onnx` each
 * carried their own `expect fun` plus six identical platform actuals, which drifted — the
 * safetensors copy returned `null` on Kotlin/Native while the GGUF copy used `pread(2)`, so the
 * same file was streamable through one loader and not the other. The per-format functions are now
 * deprecated delegates to this one.
 *
 * Returning `null` rather than throwing is deliberate: callers fall back to whole-file sequential
 * loading, which is how a browser reads a model today.
 */
public expect fun openRandomAccessSource(filePath: String): RandomAccessSource?
