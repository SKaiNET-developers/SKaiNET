package sk.ainet.int.mnist.platform

import java.io.InputStream

public actual fun readResourceBytes(path: String): ByteArray {
    val normalized = if (path.startsWith("/")) path else "/$path"
    val stream: InputStream? = object {}.javaClass.getResourceAsStream(normalized)
        ?: ResourcesJvm::class.java.getResourceAsStream(normalized)
        ?: ResourcesJvm::class.java.classLoader?.getResourceAsStream(normalized.trimStart('/'))
    return stream?.use { it.readAllBytes() }
        ?: error("Resource not found: $normalized")
}

private object ResourcesJvm