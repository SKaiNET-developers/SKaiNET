package sk.ainet.compile.c

import java.io.File

public actual fun platformCreateDirectory(path: String) {
    File(path).mkdirs()
}

public actual fun platformWriteFile(path: String, content: String) {
    val file = File(path)
    file.parentFile?.mkdirs()
    file.writeText(content)
}
