package sk.ainet.io.irpa

import kotlinx.io.Sink
import sk.ainet.lang.tensor.storage.BufferHandle

/**
 * wasmJs actual. The wasm browser sandbox has no direct filesystem
 * access, so mmap is not applicable here. Emitted `.irpa` files are
 * typically produced server-side or at build time; wasmJs consumers
 * should load pre-written archives rather than generate them. See
 * issue #523.
 */
internal actual fun writeFileBackedBytes(sink: Sink, handle: BufferHandle.FileBacked) {
    throw NotImplementedError(
        "FileBacked mmap transfer is not supported on wasmJs — no filesystem " +
            "access in the browser sandbox. Produce .irpa archives on JVM/Android " +
            "and load them at runtime."
    )
}
