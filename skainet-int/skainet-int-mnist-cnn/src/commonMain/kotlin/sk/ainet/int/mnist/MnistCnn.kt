package sk.ainet.int.mnist

/**
 * Placeholder public API to allow the module to compile.
 *
 * PRD-driven implementation will add:
 * - MnistCnnSharedState sealed interface
 * - MnistCnnShared class with Flow-based lifecycle
 * - Internal MnistCnnModule wiring to skainet-lang-models and io loaders
 */
public object MnistCnn {
    public const val MODULE_NAME: String = "skainet-int-mnist-cnn"
}