package sk.ainet.compile.c

import kotlinx.cinterop.ExperimentalForeignApi
import platform.posix.mkdir
import platform.posix.fopen
import platform.posix.fprintf
import platform.posix.fclose

@OptIn(ExperimentalForeignApi::class)
public actual fun platformCreateDirectory(path: String) {
    mkdir(path, 511.toUInt()) // 0777
}

@OptIn(ExperimentalForeignApi::class)
public actual fun platformWriteFile(path: String, content: String) {
    val file = fopen(path, "w")
    if (file != null) {
        fprintf(file, "%s", content)
        fclose(file)
    }
}
