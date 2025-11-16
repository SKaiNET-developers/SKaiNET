package sk.ainet.int.mnist.io

import java.io.IOException

public actual fun readResourceBytes(path: String): ByteArray {
    require(path.startsWith("/")) { "Resource path must start with '/': $path" }
    val stream = object {}.javaClass.getResourceAsStream(path)
        ?: throw IOException("Resource not found on classpath: $path")
    return stream.use { it.readBytes() }
}
