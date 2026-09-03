package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory, sk.ainet.context.schedule.Schedule) -> TensorOps {
    val jdkOk = isJdk21Plus()
    val vectorAvailable = jdkOk && isVectorApiAvailable()
    val useVector = (JvmCpuBackendConfig.vectorEnabled && vectorAvailable)

    if (useVector) {
        println("[SKaiNET] Using SIMD-accelerated CPU operations (Vector API)")
    } else {
        val reason = when {
            !jdkOk -> "JDK 21+ required"
            !vectorAvailable -> "Vector API not available"
            !JvmCpuBackendConfig.vectorEnabled -> "disabled by configuration"
            else -> "unknown"
        }
        println("[SKaiNET] Using standard CPU operations ($reason)")
    }

    return if (useVector) {
        { factory: TensorDataFactory, schedule: sk.ainet.context.schedule.Schedule -> DefaultCpuOpsJvm(factory, schedule) }
    } else {
        // Note: BLAS acceleration not yet implemented; falling back to DefaultCpuOps
        { factory: TensorDataFactory, schedule: sk.ainet.context.schedule.Schedule -> DefaultCpuOps(factory, schedule) }
    }
}

private fun isVectorApiAvailable(): Boolean {
    return runCatching {
        Class.forName("jdk.incubator.vector.FloatVector")
        Class.forName("jdk.incubator.vector.VectorSpecies")
        true
    }.getOrElse { false }
}

private fun isJdk21Plus(): Boolean {
    // Prefer Runtime.version() when available (Java 9+)
    val runtimeVersion = runCatching {
        val runtimeClass = Class.forName("java.lang.Runtime")
        val versionMethod = runtimeClass.getMethod("version")
        val versionObj = versionMethod.invoke(Runtime.getRuntime())
        val featureMethod = versionObj.javaClass.getMethod("feature")
        featureMethod.invoke(versionObj) as Int
    }.getOrNull()
    if (runtimeVersion != null) return runtimeVersion >= 21

    // Fallback to parsing java.specification.version
    val spec = System.getProperty("java.specification.version") ?: return false
    return spec.toIntOrNull()?.let { it >= 21 } ?: run {
        // Handle versions like "1.8", "11", "21.0.1"
        val major = spec.split('.', '-').firstOrNull()?.toIntOrNull() ?: return@run false
        major >= 21
    }
}


internal actual fun platformDefaultSchedule(): sk.ainet.context.schedule.Schedule =
    sk.ainet.exec.schedule.CoroutineSchedule.hardware()
