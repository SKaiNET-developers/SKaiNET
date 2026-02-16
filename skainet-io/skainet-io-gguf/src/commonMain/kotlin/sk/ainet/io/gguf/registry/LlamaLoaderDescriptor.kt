package sk.ainet.io.gguf.registry

/**
 * Loader descriptor for LLaMA-family models (LLaMA, Mistral, Qwen, etc.).
 *
 * Register with [ModelLoaderRegistry] to enable auto-detection of LLaMA GGUF files.
 */
public object LlamaLoaderDescriptor : ModelLoaderDescriptor {
    override val architecture: ModelArchitecture = ModelArchitecture.LLAMA
    override val displayName: String = "LLaMA"
}
