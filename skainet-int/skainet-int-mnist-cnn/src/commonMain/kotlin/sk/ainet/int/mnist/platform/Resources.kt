package sk.ainet.int.mnist.platform

/**
 * Platform-abstracted resource access.
 *
 * The GGUF model is embedded under `/models/mnist/mnist-cnn-f32.gguf`.
 */
public expect fun readResourceBytes(path: String): ByteArray

/**
 * Default resource path for the embedded MNIST CNN GGUF.
 * Matches the value documented in the PRD.
 */
public const val DEFAULT_GGUF_PATH: String = "/models/mnist/mnist-cnn-f32.gguf"
