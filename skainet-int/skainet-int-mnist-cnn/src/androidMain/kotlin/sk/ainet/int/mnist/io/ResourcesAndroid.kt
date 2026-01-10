package sk.ainet.int.mnist.io

import java.io.IOException

public actual fun readResourceBytes(path: String): ByteArray {
    require(path.startsWith("/")) { "Resource path must start with '/': $path" }
    // On Android, classloader resource access works for library packaged resources under /resources
    val cl = Thread.currentThread().contextClassLoader
        ?: ResourcesAndroid::class.java.classLoader
    val stream = (cl?.getResourceAsStream(path.removePrefix("/"))
        ?: ResourcesAndroid::class.java.getResourceAsStream(path))
        ?: throw IOException("Resource not found in Android classloader: $path")
    return stream.use { it.readBytes() }
}

private object ResourcesAndroid
