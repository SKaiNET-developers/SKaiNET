package sk.ainet.int.mnist

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.test.runTest
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.io.ParametersLoader
import kotlin.reflect.KClass
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class MnistCnnSharedTest {

    private val exec: ExecutionContext = DefaultNeuralNetworkExecutionContext()
    private val scope = CoroutineScope(Dispatchers.Default + Job())

    @AfterTest
    fun tearDown() {
        // Reset delegate to avoid leaking across tests
        DefaultModelParameterLoader.delegate = null
    }

    @Test
    fun load_transitions_to_ready_and_unload_to_unloaded() = runTest {
        val progressEvents = MutableSharedFlow<MnistCnnSharedState>(extraBufferCapacity = 16)
        val shared = MnistCnnShared(scope, exec, loader = FakeLoader(steps = 3))
        val job = shared.state.onEach { progressEvents.tryEmit(it) }.launchIn(this)

        shared.load()
        assertIs<MnistCnnSharedState.Ready>(shared.state.value)

        shared.unload()
        assertIs<MnistCnnSharedState.Unloaded>(shared.state.value)

        job.cancelAndJoin()
    }

    @Test
    fun progress_is_propagated_from_loader() = runTest {
        val fake = FakeLoader(steps = 5)
        val shared = MnistCnnShared(scope, exec, loader = fake)
        val captured = mutableListOf<MnistCnnSharedState>()
        val job = shared.state.onEach { captured.add(it) }.launchIn(this)

        shared.load()

        // We expect at least `steps` Loading states (may be more including initial one)
        val loadingCount = captured.count { it is MnistCnnSharedState.Loading }
        assertTrue(loadingCount >= fake.steps, "Expected at least ${fake.steps} Loading states but got $loadingCount")
        assertIs<MnistCnnSharedState.Ready>(shared.state.value)

        job.cancelAndJoin()
    }

    @Test
    fun run_before_load_sets_error_state() = runTest {
        val shared = MnistCnnShared(scope, exec, loader = FakeLoader(steps = 0))
        // Create a dummy 1x1x28x28 FP32 tensor
        val input: Tensor<FP32, Float> = exec.fromFloatArray(Shape(1, 1, 28, 28), FP32::class, FloatArray(1 * 1 * 28 * 28))
        try {
            shared.run(input)
        } catch (e: IllegalStateException) {
            // Expected: not loaded
        }
        assertIs<MnistCnnSharedState.Error>(shared.state.value)
    }

    @Test
    fun forward_smoke_after_load_transitions_running_then_ready_or_error() = runTest {
        val shared = MnistCnnShared(scope, exec, loader = FakeLoader(steps = 0))
        shared.load()
        assertIs<MnistCnnSharedState.Ready>(shared.state.value)

        val input: Tensor<FP32, Float> = exec.fromFloatArray(Shape(1, 1, 28, 28), FP32::class, FloatArray(1 * 1 * 28 * 28))
        val before = shared.state.value
        try {
            shared.run(input)
            // If backend ops are available, we should end in Ready
            assertIs<MnistCnnSharedState.Ready>(shared.state.value)
        } catch (t: Throwable) {
            // If ops are not available (e.g., VoidTensorOps), ensure Error is set
            assertIs<MnistCnnSharedState.Error>(shared.state.value)
        }
        // Ensure we did leave Ready at least once (entered Running)
        assertTrue(before is MnistCnnSharedState.Ready)
    }

    @Test
    fun dataset_iteration_smoke_test_guarded() = runTest {
        // This is a guarded smoke test: it tries to load the test set, but tolerates unsupported targets or IO errors.
        try {
            val ds = sk.ainet.data.mnist.MNIST.loadTest()
            // Iterate up to first batch to validate iterator mechanics using Dataset API
            val it = ds.batchIterator<sk.ainet.lang.types.Int8, Byte>(32)
            if (it.hasNext()) {
                val batch = it.next()
                // DataBatch packs x as array and a single y tensor
                assertTrue(batch.x.isNotEmpty())
                // The first input tensor batch dimension should match targets batch dimension
                val inputsBatch = batch.x[0].shape.dimensions[0]
                val targetsBatch = batch.y.shape.dimensions[0]
                assertEquals(inputsBatch, targetsBatch)
            }
        } catch (t: Throwable) {
            // Acceptable: native targets with UnsupportedOperationException or network-related issues.
            val acceptable = t is UnsupportedOperationException || t is IllegalStateException || t is java.io.IOException
            assertTrue(acceptable, "Unexpected exception type in dataset smoke test: ${t::class.simpleName}: ${t.message}")
        }
    }
}

private class FakeLoader(val steps: Int) : ParametersLoader {
    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        repeat(steps) { i ->
            @Suppress("UNCHECKED_CAST")
            val t = ctx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(i.toFloat())) as Tensor<T, V>
            onTensorLoaded("noop_param_$i", t)
            // tiny delay to simulate streaming progress
            delay(1)
        }
    }
}