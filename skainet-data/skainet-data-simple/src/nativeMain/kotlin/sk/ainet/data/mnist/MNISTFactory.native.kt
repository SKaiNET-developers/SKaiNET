package sk.ainet.data.mnist

public actual fun createMNISTLoader(config: MNISTLoaderConfig): MNISTLoader {
    throw UnsupportedOperationException("MNIST dataset download is not supported on this native target yet.")
}
