package sk.ainet.bench.publish.env

import sk.ainet.bench.publish.schema.SystemInfo
import java.io.File

public object SystemInfoProvider {

    public fun collect(): SystemInfo = SystemInfo(
        os = "${prop("os.name")} ${prop("os.version")}".trim(),
        arch = prop("os.arch"),
        cpu = readCpuModel(),
        cpuLogicalCores = Runtime.getRuntime().availableProcessors(),
        memoryGib = totalMemoryGib(),
        jdk = "${prop("java.vm.name")} ${prop("java.version")}".trim(),
        jdkVendor = prop("java.vendor"),
        ptsClient = detectPtsClient(),
    )

    private fun prop(key: String): String = System.getProperty(key, "unknown")

    private fun readCpuModel(): String {
        val cpuinfo = File("/proc/cpuinfo")
        if (cpuinfo.exists()) {
            cpuinfo.useLines { lines ->
                for (line in lines) {
                    if (line.startsWith("model name")) {
                        val v = line.substringAfter(":", "").trim()
                        if (v.isNotEmpty()) return v
                    }
                }
            }
        }
        return prop("os.arch") + " CPU"
    }

    private fun totalMemoryGib(): Long {
        val meminfo = File("/proc/meminfo")
        if (meminfo.exists()) {
            meminfo.useLines { lines ->
                for (line in lines) {
                    if (line.startsWith("MemTotal:")) {
                        val kib = line.filter { it.isDigit() }.toLongOrNull() ?: 0L
                        return kib / 1024 / 1024
                    }
                }
            }
        }
        return -1L
    }

    private fun detectPtsClient(): String? {
        val candidates = listOf("phoronix-test-suite", "pts")
        for (cmd in candidates) {
            val version = runCapture(listOf(cmd, "version")) ?: continue
            return version.lineSequence().firstOrNull { it.isNotBlank() }?.trim()
        }
        return null
    }

    private fun runCapture(command: List<String>): String? = try {
        val proc = ProcessBuilder(command)
            .redirectErrorStream(true)
            .start()
        proc.waitFor()
        if (proc.exitValue() == 0) proc.inputStream.bufferedReader().readText() else null
    } catch (_: Exception) {
        null
    }
}
