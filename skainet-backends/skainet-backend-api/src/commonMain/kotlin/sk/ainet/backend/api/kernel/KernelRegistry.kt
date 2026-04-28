package sk.ainet.backend.api.kernel

/**
 * Process-wide registry of [KernelProvider] instances.
 *
 * Backends that ship a [KernelProvider] register it via [register] at
 * load time. Callers that need a kernel ask [bestAvailable] (or
 * [find] with a name) for the highest-priority provider that reports
 * itself available, then pull the specific kernel they need from the
 * provider's accessors.
 *
 * The registry is plain manual registration today — JVM auto-discovery
 * via `java.util.ServiceLoader` can be layered on in a follow-up PR
 * once a second concrete provider exists (Panama Vector). Callers that
 * want a guaranteed scalar fallback can pin
 * `sk.ainet.exec.kernel.ScalarKernelProvider` directly without going
 * through the registry.
 *
 * Thread safety: [register] is not thread-safe. Call it during
 * single-threaded startup or guard with your own lock.
 */
public object KernelRegistry {
    private val providers: MutableList<KernelProvider> = mutableListOf()

    /**
     * Register a provider. Re-registering the same instance is a no-op.
     */
    public fun register(provider: KernelProvider) {
        if (providers.any { it === provider }) return
        providers.add(provider)
        providers.sortByDescending { it.priority }
    }

    /** All registered providers, sorted by priority descending. */
    public fun providers(): List<KernelProvider> = providers.toList()

    /** Find a provider by name (case-insensitive), or `null`. */
    public fun find(name: String): KernelProvider? =
        providers.firstOrNull { it.name.equals(name, ignoreCase = true) }

    /**
     * Highest-priority [isAvailable] provider, or `null` if none is
     * registered or available. Callers that absolutely need a kernel
     * should use the explicit scalar fallback instead.
     */
    public fun bestAvailable(): KernelProvider? =
        providers.firstOrNull { it.isAvailable() }

    /** Names of all currently-available providers. */
    public fun availableNames(): List<String> =
        providers.filter { it.isAvailable() }.map { it.name }

    /** Test/diagnostic helper. Removes all registered providers. */
    public fun clearForTesting() {
        providers.clear()
    }
}
