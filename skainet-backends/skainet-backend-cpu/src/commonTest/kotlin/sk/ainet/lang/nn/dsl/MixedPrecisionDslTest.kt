package sk.ainet.lang.nn.dsl

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.PrecisionBoundary
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * SKEEP-008 phase A1 (#1433): `dense(dtype = …)` and `stage(id, dtype = …)` run their layers
 * in another dtype behind a [PrecisionBoundary], casting in and out with `TensorOps.convert`.
 */
class MixedPrecisionDslTest {

    private val ctx = DirectCpuExecutionContext()

    // Values deliberately not representable in FP16/BF16, so the boundary's rounding is observable.
    private val w1 = floatArrayOf(
        0.1234567f, -1.7654321f,
        2.3456789f, 0.0123456f,
        -0.9876543f, 1.1111111f,
    )
    private val b1 = floatArrayOf(0.1111111f, -0.2222222f, 0.3333333f)
    private val w2 = floatArrayOf(
        1.9999999f, -0.1111111f, 0.5555555f,
        -1.3333333f, 0.7777777f, -0.4444444f,
    )
    private val b2 = floatArrayOf(0.0101010f, -0.0202020f)

    private fun input(): Tensor<FP32, Float> = data<FP32, Float>(ctx) {
        tensor<FP32, Float> {
            shape(1, 2) { fromArray(floatArrayOf(0.3f, -1.1f)) }
        }
    }

    private fun fp32Reference(context: ExecutionContext): Module<FP32, Float> = definition {
        network<FP32, Float>(context) {
            input(2)
            dense(3, "layer1") {
                weights { fromArray(w1) }
                bias { fromArray(b1) }
            }
            dense(2, "layer2") {
                weights { fromArray(w2) }
                bias { fromArray(b2) }
            }
        }
    }

    @Test
    fun denseWithDtype_runsAndRoundsThroughBf16() {
        val mixed = definition {
            network<FP32, Float>(ctx) {
                input(2)
                dense(3, dtype = BF16, id = "layer1") {
                    weights { fromArray(w1) }
                    bias { fromArray(b1) }
                }
                dense(2, "layer2") {
                    weights { fromArray(w2) }
                    bias { fromArray(b2) }
                }
            }
        }

        val reference = fp32Reference(ctx).forward(input(), ctx)
        val result = mixed.forward(input(), ctx)

        assertEquals(Shape(1, 2), result.shape)
        assertEquals(FP32::class, result.dtype, "the boundary must cast back to the network dtype")
        var maxDiff = 0f
        for (c in 0 until 2) {
            val diff = abs(result.data[0, c] - reference.data[0, c])
            assertTrue(diff <= 2e-2f, "BF16 layer drifted too far at $c: diff=$diff")
            maxDiff = maxOf(maxDiff, diff)
        }
        assertTrue(maxDiff > 0f, "BF16 rounding must change the result, or the cast never happened")
    }

    @Test
    fun stageWithDtype_runsAndRoundsThroughFp16() {
        val mixed = definition {
            network<FP32, Float>(ctx) {
                input(2)
                stage("body", dtype = FP16) {
                    dense(3, "layer1") {
                        weights { fromArray(w1) }
                        bias { fromArray(b1) }
                    }
                    dense(2, "layer2") {
                        weights { fromArray(w2) }
                        bias { fromArray(b2) }
                    }
                }
            }
        }

        val reference = fp32Reference(ctx).forward(input(), ctx)
        val result = mixed.forward(input(), ctx)

        assertEquals(Shape(1, 2), result.shape)
        assertEquals(FP32::class, result.dtype, "the boundary must cast back to the network dtype")
        var maxDiff = 0f
        for (c in 0 until 2) {
            val diff = abs(result.data[0, c] - reference.data[0, c])
            assertTrue(diff <= 2e-2f, "FP16 stage drifted too far at $c: diff=$diff")
            maxDiff = maxOf(maxDiff, diff)
        }
        assertTrue(maxDiff > 0f, "FP16 rounding must change the result, or the cast never happened")
    }

    @Test
    fun sameDtypeBoundary_isExact() {
        // convert returns its input when the dtype matches, so an FP32 stage inside an
        // FP32 network reproduces the reference bit-exactly.
        val wrapped = definition {
            network<FP32, Float>(ctx) {
                input(2)
                stage("body", dtype = FP32) {
                    dense(3, "layer1") {
                        weights { fromArray(w1) }
                        bias { fromArray(b1) }
                    }
                    dense(2, "layer2") {
                        weights { fromArray(w2) }
                        bias { fromArray(b2) }
                    }
                }
            }
        }

        val reference = fp32Reference(ctx).forward(input(), ctx)
        val result = wrapped.forward(input(), ctx)

        for (c in 0 until 2) {
            assertEquals(reference.data[0, c], result.data[0, c], "FP32 boundary must be a no-op at $c")
        }
    }

    @Test
    fun mixedLayers_appearAsPrecisionBoundaryModules() {
        val mixed = definition {
            network<FP32, Float>(ctx) {
                input(2)
                dense(3, dtype = BF16, id = "layer1") {
                    weights { fromArray(w1) }
                    bias { fromArray(b1) }
                }
            }
        }

        val boundaries = mixed.modules.filterIsInstance<PrecisionBoundary<*, *, *>>()
        assertEquals(1, boundaries.size, "the BF16 dense must be wrapped in one PrecisionBoundary")
        assertEquals("layer1", boundaries.single().name)
    }
}
