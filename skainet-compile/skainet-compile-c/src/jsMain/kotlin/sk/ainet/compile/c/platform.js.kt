package sk.ainet.compile.c

public actual fun platformCreateDirectory(path: String) {
    // No-op for JS/Wasm in browser
}

public actual fun platformWriteFile(path: String, content: String) {
    // No-op for JS/Wasm in browser
}
