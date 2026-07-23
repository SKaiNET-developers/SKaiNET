package sk.ainet.lang.nn.topology

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


/**
 * Lightweight wrapper for a trainable parameter.
 * Used for optimizer registration and tracking.
 */
public data class Parameter(
    public val name: String,
    public val moduleParameter: ModuleParameter<*, *>
) {
    public val id: String get() = moduleParameter.value.gradState.toString() // Or use a proper unique ID if available
    public val value: Tensor<*, *> get() = moduleParameter.value
    public val requiresGrad: Boolean get() = moduleParameter.requiresGrad
}

public sealed class ModuleParameter<T : DType, V>(
    public val requiresGrad: Boolean = true
) {
    public abstract val name: String
    public abstract var value: Tensor<T, V>

    protected fun syncFlag() {
        value.gradState.requiresGrad = requiresGrad
    }

    public data class WeightParameter<T : DType, V>(
        override val name: String,
        private var backingValue: Tensor<T, V>,
        public val trainable: Boolean = true
    ) : ModuleParameter<T, V>(trainable) {
        override var value: Tensor<T, V>
            get() = backingValue
            set(v) {
                backingValue = v
                syncFlag()
            }

        init {
            syncFlag()
        }
    }

    public data class BiasParameter<T : DType, V>(
        override val name: String,
        private var backingValue: Tensor<T, V>,
        public val trainable: Boolean = true
    ) : ModuleParameter<T, V>(trainable) {
        override var value: Tensor<T, V>
            get() = backingValue
            set(v) {
                backingValue = v
                syncFlag()
            }

        init {
            syncFlag()
        }
    }
}

public interface ModuleParameters<T : DType, V> {
    public val params: List<ModuleParameter<T, V>>
}

public fun <T : DType, V> List<ModuleParameter<T, V>>.by(name: String): ModuleParameter<T, V>? =
    firstOrNull { namedParameter -> namedParameter.name.uppercase().contains(name.uppercase()) }

// Returns the first BiasParameter or throws a NoSuchElementException if none is found.
public fun <T : DType, V> List<ModuleParameter<T, V>>.bias(): ModuleParameter.BiasParameter<T, V> =
    this.filterIsInstance<ModuleParameter.BiasParameter<T, V>>()
        .firstOrNull() ?: throw NoSuchElementException("No bias parameter found!")

// Returns the first BiasParameter or null for modules without a bias (e.g. Linear(bias = null)).
public fun <T : DType, V> List<ModuleParameter<T, V>>.biasOrNull(): ModuleParameter.BiasParameter<T, V>? =
    this.filterIsInstance<ModuleParameter.BiasParameter<T, V>>().firstOrNull()

// Returns the first WeightParameter or throws a NoSuchElementException if none is found.
public fun <T : DType, V> List<ModuleParameter<T, V>>.weights(): ModuleParameter.WeightParameter<T, V> =
    this.filterIsInstance<ModuleParameter.WeightParameter<T, V>>()
        .firstOrNull() ?: throw NoSuchElementException("No weight parameter found!")

/**
 * Zero accumulated gradients for a collection of module parameters.
 */
public fun Iterable<ModuleParameter<*, *>>.zeroGrad() {
    for (p in this) {
        p.value.zeroGrad()
    }
}
