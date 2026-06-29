package sk.ainet.data.common

/**
 * Supplies a Hugging Face token for built-in dataset loaders when their source
 * URIs point at private Hugging Face artifacts.
 */
public fun interface DatasetHuggingFaceTokenProvider {
    public fun token(): String?
}
