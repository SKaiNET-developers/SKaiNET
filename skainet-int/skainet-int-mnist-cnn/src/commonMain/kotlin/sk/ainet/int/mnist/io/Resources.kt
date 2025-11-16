package sk.ainet.int.mnist.io

/**
 * Cross-platform resource access for the embedded MNIST CNN model and other assets.
 *
 * The path should be an absolute resource path starting with '/'.
 * Example: DEFAULT_GGUF_PATH
 */
public expect fun readResourceBytes(path: String): ByteArray

/** Default embedded GGUF resource path for the MNIST CNN model. */
public const val DEFAULT_GGUF_PATH: String = "/models/mnist/mnist-cnn-f32.gguf"
