package sk.ainet.int.mnist

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.load
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.topology.by
import kotlin.reflect.KClass

// --- Public API (as specified in PRD mnist-int.md) ---

sealed interface MnistCnnSharedState {
    data object Unloaded : MnistCnnSharedState
    data class Loading(val current: Long, val total: Long, val message: String?) : MnistCnnSharedState
    data class Ready(val module: MnistCnnModule) : MnistCnnSharedState
    data class Running(val current: Long, val total: Long, val message: String?) : MnistCnnSharedState
    data class Error(val throwable: Throwable) : MnistCnnSharedState
}

/**
 * Type alias matching the PRD nomenclature while reusing existing io-core abstraction.
 */
public typealias ModelParameterLoader = ParametersLoader

/**
 * Minimal placeholder for the internal model module referenced by the public state.
 * Actual wiring and forward pass will be implemented in task "3. Internal Assembly".
 */
public class MnistCnnModule internal constructor(
    val execContext: ExecutionContext,
) {
    private val model = sk.ainet.lang.model.dnn.cnn.MnistCnn().create(execContext)

    suspend fun loadParameters(
        loader: ModelParameterLoader,
        onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> }
    ) {
        var current = 0L
        // Load FP32 tensors and assign by best-effort name matching into the model parameters
        loader.load<FP32, Float>(execContext) { name, tensor ->
            applyTensorByName(name, tensor)
            current += 1
            onProgress(current, 0, name)
        }
    }

    private fun applyTensorByName(name: String, tensor: Tensor<FP32, Float>) {
        fun visitAssign(m: sk.ainet.lang.nn.Module<FP32, Float>): Boolean {
            val paramsHolder = m as? sk.ainet.lang.nn.topology.ModuleParameters<FP32, Float>
            if (paramsHolder != null) {
                val target = paramsHolder.params.by(name)
                if (target != null) {
                    target.value = tensor
                    return true
                }
            }
            for (child in m.modules) {
                if (visitAssign(child)) return true
            }
            return false
        }
        visitAssign(model)
    }

    suspend fun forward(input: Tensor<FP32, Float>): Tensor<FP32, Float> = model.forward(input, execContext)
}

class MnistCnnShared(
    private val scope: CoroutineScope,
    private val execContext: ExecutionContext,
    private val loader: ModelParameterLoader = DefaultModelParameterLoader,
) {
    private val _state = MutableStateFlow<MnistCnnSharedState>(MnistCnnSharedState.Unloaded)
    val state: StateFlow<MnistCnnSharedState> = _state

    private val guard = Mutex()
    private var module: MnistCnnModule? = null

    suspend fun load(resourcePath: String = DEFAULT_GGUF_PATH) {
        guard.withLock {
            try {
                // Emit a single-step loading progress; real progress comes with loader integration (task 4)
                _state.value = MnistCnnSharedState.Loading(current = 0, total = 1, message = "Initializing")

                // Build module and load parameters via loader
                val m = MnistCnnModule(execContext)
                module = m
                try {
                    m.loadParameters(loader) { current, total, message ->
                        _state.value = MnistCnnSharedState.Loading(current, if (total == 0L) 100 else total, message)
                    }
                } catch (ce: CancellationException) {
                    // Preserve cooperative cancellation without converting to Error
                    module = null
                    _state.value = MnistCnnSharedState.Unloaded
                    throw ce
                } catch (t: Throwable) {
                    _state.value = MnistCnnSharedState.Error(t)
                    throw t
                }

                // Mark loaded
                _state.value = MnistCnnSharedState.Ready(m)
            } catch (ce: CancellationException) {
                // Do not override cancellation with Error state; leave as set above
                throw ce
            } catch (t: Throwable) {
                _state.value = MnistCnnSharedState.Error(t)
            }
        }
    }

    suspend fun run(input: Tensor<FP32, Float>): Tensor<FP32, Float> = guard.withLock {
        val m = module ?: run {
            val ex = IllegalStateException("MNIST CNN not loaded. Call load() first.")
            _state.value = MnistCnnSharedState.Error(ex)
            throw ex
        }
        return@withLock try {
            _state.value = MnistCnnSharedState.Running(current = 0, total = 1, message = "Running inference")
            val out = m.forward(input)
            _state.value = MnistCnnSharedState.Ready(m)
            out
        } catch (ce: CancellationException) {
            // Restore to Ready on cooperative cancellation and propagate
            _state.value = MnistCnnSharedState.Ready(m)
            throw ce
        } catch (t: Throwable) {
            _state.value = MnistCnnSharedState.Error(t)
            throw t
        }
    }

    suspend fun unload() {
        guard.withLock {
            module = null
            _state.value = MnistCnnSharedState.Unloaded
        }
    }

    companion object {
        const val DEFAULT_GGUF_PATH: String = "/models/mnist/mnist-cnn-f32.gguf"
    }
}

/**
 * Default loader placeholder. It keeps the API surface compiling without yet wiring
 * the actual GGUF-backed implementation (covered by later tasks). Any invocation
 * will currently be a no-op.
 */
object DefaultModelParameterLoader : ModelParameterLoader {
    // Delegates to a concrete loader (e.g., io-gguf) provided by the integration layer.
    // This keeps the public API independent from concrete IO types (like kotlinx.io.Source).
    @Volatile
    var delegate: ModelParameterLoader? = null

    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        val d = delegate ?: error("DefaultModelParameterLoader: delegate not set. Provide an io-gguf-backed ParametersLoader before calling load().")
        return d.load(ctx, dtype, onTensorLoaded)
    }
}
