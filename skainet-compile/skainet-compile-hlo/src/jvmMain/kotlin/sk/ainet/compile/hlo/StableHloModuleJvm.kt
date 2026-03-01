package sk.ainet.compile.hlo

import java.io.InputStream
import java.io.Reader
import java.io.StringReader

/**
 * Returns a [Reader] over the MLIR [content][StableHloModule.content].
 */
public fun StableHloModule.asReader(): Reader = StringReader(content)

/**
 * Returns an [InputStream] over the MLIR [content][StableHloModule.content] encoded as UTF-8.
 */
public fun StableHloModule.asInputStream(): InputStream = content.byteInputStream(Charsets.UTF_8)
