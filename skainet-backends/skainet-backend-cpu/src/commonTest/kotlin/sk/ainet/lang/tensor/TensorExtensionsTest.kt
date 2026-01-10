package sk.ainet.lang.tensor

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.math.sqrt

class TensorExtensionsTest {

    private val executionContext = DirectCpuExecutionContext()

    private fun <T : sk.ainet.lang.types.DType> Tensor<T, Float>.item(vararg indices: Int): Float {
        return if (rank == 0 && indices.size == 1 && indices[0] == 0) {
            data.get()
        } else {
            data.get(*indices)
        }
    }

    @Test
    fun testCosineDistance() {
        data(executionContext) {
            val a = tensor<FP32, Float> {
                shape(3) {
                    from(1.0f, 0.0f, 0.0f)
                }
            }
            val b = tensor<FP32, Float> {
                shape(3) {
                    from(0.0f, 1.0f, 0.0f)
                }
            }
            val c = tensor<FP32, Float> {
                shape(3) {
                    from(1.0f, 1.0f, 0.0f)
                }
            }

            // Cosine distance between [1, 0, 0] and [0, 1, 0] is 1 - 0 = 1
            val distAB = a.cosineDistance(b)
            assertEquals(1.0f, distAB.item(0), 1e-6f)

            // Cosine similarity between [1, 0, 0] and [1, 1, 0] is 1 / (1 * sqrt(2)) = 0.70710678
            // Cosine distance is 1 - 0.70710678 = 0.29289322
            val distAC = a.cosineDistance(c)
            val expectedAC = 1.0f - (1.0f / sqrt(2.0f))
            assertEquals(expectedAC.toFloat(), distAC.item(0), 1e-6f)

            // Cosine distance with itself should be 0 (or very close to it due to epsilon)
            val distAA = a.cosineDistance(a)
            // dot(a,a)=1, norm(a)=1, norm(a)=1, similarity = 1/(1*1 + eps) approx 1
            assertEquals(0.0f, distAA.item(0), 1e-4f) // Larger tolerance due to eps
        }
    }

    @Test
    fun testCosineDistanceWithZeroVector() {
        data(executionContext) {
            val a = tensor<FP32, Float> {
                shape(3) {
                    from(1.0f, 2.0f, 3.0f)
                }
            }
            val zero = tensor<FP32, Float> {
                shape(3) {
                    zeros()
                }
            }

            // With zero vector, denominator is 0 * norm(a) + eps = eps
            // Numerator is dot(a, zero) = 0
            // Cosine similarity is 0 / eps = 0
            // Cosine distance is 1 - 0 = 1
            val dist = a.cosineDistance(zero)
            assertEquals(1.0f, dist.item(0), 1e-6f)
        }
    }

    @Test
    fun testCosineDistanceAlongDimension() {
        data(executionContext) {
            // 2 vectors of size 3
            val a = tensor<FP32, Float> {
                shape(2, 3) {
                    from(
                        1.0f, 0.0f, 0.0f,
                        1.0f, 1.0f, 0.0f
                    )
                }
            }
            val b = tensor<FP32, Float> {
                shape(2, 3) {
                    from(
                        0.0f, 1.0f, 0.0f,
                        1.0f, 1.0f, 0.0f
                    )
                }
            }

            val dist = a.cosineDistance(b, dim = -1)
            assertEquals(2, dist.volume)
            assertEquals(1.0f, dist.item(0), 1e-6f) // dist([1,0,0], [0,1,0]) = 1
            assertEquals(0.0f, dist.item(1), 1e-4f) // dist([1,1,0], [1,1,0]) = 0
        }
    }
}
