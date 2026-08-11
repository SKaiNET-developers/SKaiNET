package sk.ainet.exec.kernel.jni

import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5KMatmulKernel
import sk.ainet.backend.api.kernel.Q5_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5_1MatmulKernel
import sk.ainet.backend.api.kernel.Q6KMatmulKernel
import sk.ainet.backend.api.kernel.Q8_0MatmulKernel
import sk.ainet.backend.api.kernel.Fp32MatmulKernel

/**
 * Priority-100 [KernelProvider] backed by the JNI bridge to the shared C
 * kernels — the Android counterpart of the JVM's FFM `native-ffm` provider
 * (ART has no `java.lang.foreign`, so FFM can never run there).
 *
 * Availability is probed once: the two-tier loader in [JniKernels] must have
 * loaded a library variant AND the smoke kernel must round-trip correctly.
 * Every failure mode is non-fatal — the registry then cascades to the
 * scalar provider, exactly like the JVM behaves on hosts where the FFM lib
 * doesn't load.
 *
 * Discovered via `META-INF/services` ([JniKernelProviderFactory]) by the
 * ServiceLoader install in `skainet-backend-cpu`'s Android ops factory.
 */
public object JniKernelProvider : KernelProvider {

    override val name: String = "native-jni"

    override val priority: Int = 100

    /** Which library tier actually loaded — for diagnostics/field reports. */
    public val activeVariant: JniKernels.Variant? get() = JniKernels.variant

    private val available: Boolean by lazy {
        if (!JniKernels.isLoaded) return@lazy false
        runCatching {
            val input = floatArrayOf(1.0f, 2.5f, -3.0f)
            val output = FloatArray(3)
            JniKernels.smoke(input, output, 3)
            output[0] == 2.0f && output[1] == 5.0f && output[2] == -6.0f
        }.getOrDefault(false)
    }

    override fun isAvailable(): Boolean = available

    override fun matmulFp32(): Fp32MatmulKernel? = null // GEMM shim not bridged yet (#920)

    override fun matmulQ8_0(): Q8_0MatmulKernel? = if (available) JniQ8_0Matmul else null

    override fun matmulQ4_0(): Q4_0MatmulKernel? = if (available) JniQ4_0Matmul else null

    override fun matmulQ4K(): Q4KMatmulKernel? = if (available) JniQ4KMatmul else null

    override fun matmulQ5K(): Q5KMatmulKernel? = if (available) JniQ5KMatmul else null

    override fun matmulQ6K(): Q6KMatmulKernel? = if (available) JniQ6KMatmul else null

    override fun matmulQ5_0(): Q5_0MatmulKernel? = if (available) JniQ5_0Matmul else null

    override fun matmulQ5_1(): Q5_1MatmulKernel? = if (available) JniQ5_1Matmul else null

    private object JniQ8_0Matmul : Q8_0MatmulKernel {
        override fun matmul(
            input: FloatArray, inputOffset: Int,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            output: FloatArray, outputOffset: Int,
        ): Unit = JniKernels.q80Matmul(
            input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset
        )
    }

    private object JniQ4_0Matmul : Q4_0MatmulKernel {
        override fun matmul(
            input: FloatArray, inputOffset: Int,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            output: FloatArray, outputOffset: Int,
        ): Unit = JniKernels.q40Matmul(
            input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset
        )
    }

    private object JniQ5_0Matmul : Q5_0MatmulKernel {
        override fun matmul(
            input: FloatArray, inputOffset: Int,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            output: FloatArray, outputOffset: Int,
        ): Unit = JniKernels.q50Matmul(
            input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset
        )
    }

    private object JniQ5_1Matmul : Q5_1MatmulKernel {
        override fun matmul(
            input: FloatArray, inputOffset: Int,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            output: FloatArray, outputOffset: Int,
        ): Unit = JniKernels.q51Matmul(
            input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset
        )
    }

    private object JniQ4KMatmul : Q4KMatmulKernel {
        override fun matmul(
            input: FloatArray, inputOffset: Int,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            output: FloatArray, outputOffset: Int,
        ): Unit = JniKernels.q4kMatmul(
            input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset
        )
    }

    private object JniQ5KMatmul : Q5KMatmulKernel {
        override fun matmul(
            input: FloatArray, inputOffset: Int,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            output: FloatArray, outputOffset: Int,
        ): Unit = JniKernels.q5kMatmul(
            input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset
        )
    }

    private object JniQ6KMatmul : Q6KMatmulKernel {
        override fun matmul(
            input: FloatArray, inputOffset: Int,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            output: FloatArray, outputOffset: Int,
        ): Unit = JniKernels.q6kMatmul(
            input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset
        )
    }
}

/**
 * `ServiceLoader`-friendly wrapper around [JniKernelProvider]: the service
 * machinery requires a public no-arg constructor, which a Kotlin `object`
 * does not expose. Mirrors `NativeKernelProviderFactory` on the JVM side.
 *
 * Listed in `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider`;
 * `consumer-rules.pro` keeps both the entry and this class through R8.
 */
public class JniKernelProviderFactory : KernelProvider by JniKernelProvider
