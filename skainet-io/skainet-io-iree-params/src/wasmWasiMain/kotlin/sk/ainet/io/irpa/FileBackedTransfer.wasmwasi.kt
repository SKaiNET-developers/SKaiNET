package sk.ainet.io.irpa

import kotlinx.io.Sink
import sk.ainet.lang.tensor.storage.BufferHandle

/**
 * wasmWasi actual. WASI has filesystem primitives but no mmap syscall
 * today (the preview2 proposal adds one but isn't widespread yet).
 * Same deferral rationale as the Native target. See issue #523.
 */
internal actual fun writeFileBackedBytes(sink: Sink, handle: BufferHandle.FileBacked) {
    throw NotImplementedError(
        "FileBacked mmap transfer is not yet implemented on wasmWasi. See " +
            "issue #523 PR E follow-up."
    )
}
