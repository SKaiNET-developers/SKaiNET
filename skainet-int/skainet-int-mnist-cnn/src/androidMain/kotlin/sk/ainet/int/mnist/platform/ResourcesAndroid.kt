package sk.ainet.int.mnist.platform

import android.content.Context

// A ContentProvider-like context holder to access app context when available.
// For library packaging, we attempt to read from the classloader resources first.
@Suppress("UnusedReceiverParameter")
public actual fun readResourceBytes(path: String): ByteArray {
    val normalized = if (path.startsWith("/")) path else "/$path"
    val cl = ResourcesAndroid::class.java.classLoader
    val fromClassLoader = cl?.getResourceAsStream(normalized.trimStart('/'))
        ?: ResourcesAndroid::class.java.getResourceAsStream(normalized)
    if (fromClassLoader != null) {
        return fromClassLoader.use { it.readBytes() }
    }
    // Fallback: if a Context is injected in future, could load from assets.
    error("Resource not found on Android: $normalized. Ensure it is packaged under src/commonMain/resources or src/androidMain/resources.")
}

private object ResourcesAndroid