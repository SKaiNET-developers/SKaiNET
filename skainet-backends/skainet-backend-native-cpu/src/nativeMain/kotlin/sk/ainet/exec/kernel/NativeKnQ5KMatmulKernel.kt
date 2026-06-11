package sk.ainet.exec.kernel

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import sk.ainet.backend.api.kernel.Q5KMatmulKernel
import sk.ainet.kernels.cinterop.skainet_q5k_matmul

/**
 * Kotlin/Native implementation of [Q5KMatmulKernel] that calls the hand-written
 * C kernel `skainet_q5k_matmul` (the same `q5k_matmul.c` the JVM consumes via
 * FFM) through cinterop, linked from the static archive `libskainet_kernels.a`.
 *
 * This is the board-consumption path: the SL2610 binary is Kotlin/Native, so it
 * cannot use the JVM-FFM wrapper. The arrays are pinned and their base pointers
 * passed to C; the C side reads `input + input_offset` etc., so no copy is made.
 *
 * On `linuxArm64` the linked archive carries the NEON paths
 * (`-march=armv8.2-a+fp16+dotprod`); on `linuxX64` (this POC host) it's the
 * scalar/auto-vectorized build. Correctness is identical across both.
 */
@OptIn(ExperimentalForeignApi::class)
public object NativeKnQ5KMatmulKernel : Q5KMatmulKernel {

    private const val BLOCK_SIZE = 256

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeKnQ5KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return

        input.usePinned { inPin ->
            weight.usePinned { wPin ->
                output.usePinned { outPin ->
                    skainet_q5k_matmul(
                        inPin.addressOf(0),
                        inputOffset,
                        wPin.addressOf(0).reinterpret(),
                        weightByteOffset,
                        inputDim,
                        outputDim,
                        outPin.addressOf(0),
                        outputOffset,
                    )
                }
            }
        }
    }
}
