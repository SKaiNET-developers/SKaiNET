package sk.ainet.exec.autograd

import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.dsl.skainet
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.*
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SkainetScopeTest {

    private fun createTrainCtx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        val cpuOps = DefaultCpuOps(dataFactory)
        val graph = sk.ainet.lang.graph.DefaultComputeGraph()
        return DefaultGraphExecutionContext(
            baseOps = cpuOps,
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) },
            computeGraph = graph,
            baseSink = sk.ainet.lang.trace.GraphSink(graph)
        )
    }

    private fun createEvalCtx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        val cpuOps = DefaultCpuOps(dataFactory)
        return DefaultGraphExecutionContext(
            baseOps = cpuOps,
            phase = Phase.EVAL,
            tensorDataFactory = dataFactory
        )
    }

    @Test
    fun `test SkainetScope DSL flow`() {
        skainet(createTrainCtx()) {
            val a = tensor<FP32, Float> {
                shape(3) {
                    from(1f, 0f, 0f)
                }
            }.withRequiresGrad()
            val b = tensor<FP32, Float> {
                shape(3) {
                    from(0f, 1f, 0f)
                }
            }

            val aParam = ModuleParameter.WeightParameter("a", a)
            val initialVal1 = aParam.value.data.get(1)

            // Use trainStep
            trainStep(sgd(lr = 0.5), aParam) {
                aParam.value.cosineDistance(b)
            }

            val updatedVal1 = aParam.value.data[1]
            
            assertTrue(updatedVal1 != initialVal1, "Parameter 'a' should have been updated")
            assertEquals(aParam.value.grad, null, "Gradients should have been zeroed")
        }
    }

    @Test
    fun `test plain inference DSL`() {
        // Plain inference doesn't need a tape or training context
        skainet(createEvalCtx()) {
            val a = tensor<FP32, Float> {
                shape(3) {
                    from(1f, 2f, 3f)
                }
            }

            val b = tensor<FP32, Float> {
                shape(3) {
                    from(4f, 5f, 6f)
                }
            }

            // Direct usage without record/trainStep
            val result = a + b

            assertEquals(result.shape, Shape(3))
            assertEquals(result.data[0], 5f)
            assertEquals(result.data[1], 7f)
            assertEquals(result.data[2], 9f)
        }
    }

    @Test
    fun `test simple MLP training mimic`() {
        skainet(createTrainCtx()) {
            // Mocking a simple MLP: y = relu(x * w + b)
            val x = ctx.fromFloatArray<FP32, Float>(Shape(1, 2), FP32::class, floatArrayOf(1f, 2f))
            
            // Parameters: (2, 2)
            val w = ctx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)).withRequiresGrad()
            val b = ctx.fromFloatArray<FP32, Float>(Shape(1, 2), FP32::class, floatArrayOf(0.1f, 0.1f)).withRequiresGrad()
            
            val wParam = ModuleParameter.WeightParameter("w", w)
            val bParam = ModuleParameter.BiasParameter("b", b)
            
            val target = ctx.fromFloatArray<FP32, Float>(Shape(1, 2), FP32::class, floatArrayOf(1f, 1f))
            val optimizer = sgd(lr = 0.1)

            // Initial loss
            val initialLoss = ((((x.matmul(wParam.value) + bParam.value).relu()) - target).let { it * it }).sum()
            val initialLossVal = initialLoss.data.get()

            // One training step
            trainStep(optimizer, wParam, bParam) {
                val layer1 = (x.matmul(wParam.value) + bParam.value).relu()
                val diff = layer1 - target
                (diff * diff).sum()
            }

            // Verify update
            val finalLoss = ((((x.matmul(wParam.value) + bParam.value).relu()) - target).let { it * it }).sum()
            val finalLossVal = finalLoss.data.get()
            
            assertTrue(finalLossVal < initialLossVal || finalLossVal == 0f, "Loss should decrease ($initialLossVal -> $finalLossVal)")
            assertTrue(wParam.value.grad == null, "Gradients should be cleared")
        }
    }

    @Test
    fun `test new tensor DSL in SkainetScope`() {
        skainet(createEvalCtx()) {
            // Using the new DSL to create Int32 tensor
            val intTensor = tensor<Int32, Int> {
                shape(2, 2) {
                    from(1, 2, 3, 4)
                }
            }

            assertEquals(Shape(2, 2), intTensor.shape)
            assertEquals(1, intTensor.data[0, 0])
            assertEquals(4, intTensor.data[1, 1])

            // Using zeros/ones from the DSL
            val zeros = tensor<FP32, Float> {
                shape(3) { zeros() }
            }
            assertEquals(0f, zeros.data[0])
            assertEquals(Shape(3), zeros.shape)

            val ones = tensor<FP32, Float> {
                shape(2, 1) { ones() }
            }
            assertEquals(1f, ones.data[0, 0])
            assertEquals(1f, ones.data[1, 0])
            assertEquals(Shape(2, 1), ones.shape)

            // Using scalar
            val s = scalar(42, Int32::class)
            assertEquals(42, s.run { data.get() })
            assertEquals(Shape(), s.shape)
        }
    }
}
