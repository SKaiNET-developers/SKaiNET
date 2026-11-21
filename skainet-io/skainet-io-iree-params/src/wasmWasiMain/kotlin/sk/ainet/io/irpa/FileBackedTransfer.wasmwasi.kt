package sk.ainet.io.irpa

import kotlinx.io.Sink
import sk.ainet.compile.hlo.ConstantSource

/**
 * wasmWasi actual. WASI has filesystem primitives but no mmap syscall
 * today (the preview2 proposal adds one but isn't widespread yet).
 * Same deferral rationale as the Native target. See issue #523.
 */
internal actual fun writeFileRangeBytes(sink: Sink, handle: ConstantSource.FileRange) {
    throw NotImplementedError(
        "FileBacked mmap transfer is not yet implemented on wasmWasi. See " +
            "issue #523 PR E follow-up."
    )
}
