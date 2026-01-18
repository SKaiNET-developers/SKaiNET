package sk.ainet.data.fashionmnist

/**
 * Linux implementation of the Fashion-MNIST loader factory.
 */
public actual fun createFashionMNISTLoader(config: FashionMNISTLoaderConfig): FashionMNISTLoader {
    return FashionMNISTLoaderLinux(config)
}
