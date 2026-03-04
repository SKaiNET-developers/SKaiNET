package sk.ainet.io.gguf.registry

import sk.ainet.io.model.ModelArchitecture

/**
 * Loader descriptor for Gemma-family models (Gemma, Gemma 3, Gemma 3n).
 *
 * Register with [ModelLoaderRegistry] to enable auto-detection of Gemma GGUF files.
 */
public object Gemma3nLoaderDescriptor : ModelLoaderDescriptor {
    override val architecture: ModelArchitecture = ModelArchitecture.GEMMA
    override val displayName: String = "Gemma 3n"
}
