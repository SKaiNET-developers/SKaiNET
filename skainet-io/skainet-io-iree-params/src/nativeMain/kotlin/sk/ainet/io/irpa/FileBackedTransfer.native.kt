package sk.ainet.io.irpa

import kotlinx.io.Sink
import sk.ainet.compile.hlo.ConstantSource

/**
 * Native actual for [writeFileBackedBytes]. Kotlin/Native can reach
 * `mmap(2)` via cinterop, but wiring that up cleanly across every
 * native target (iosArm64, iosSimulatorArm64, macosArm64, linuxX64,
 * linuxArm64) is deferred — PR E focuses on the JVM + Android path
 * which is where real inference workloads run today.
 *
 * Callers that hit this on native should either read the range into
 * a byte array ([ConstantSource.Bytes]) before handing it to
 * [IrpaWriter], or wait for native mmap support.
 */
internal actual fun writeFileRangeBytes(sink: Sink, handle: ConstantSource.FileRange) {
    throw NotImplementedError(
        "FileBacked mmap transfer is not yet implemented on Kotlin/Native. " +
            "See issue #523 PR E follow-up. Read the range into a byte array " +
            "(ConstantSource.Bytes) before writing."
    )
}
