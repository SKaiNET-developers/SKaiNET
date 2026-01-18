package sk.ainet.data.fashionmnist

/**
 * macOS implementation of the Fashion-MNIST loader factory.
 */
public actual fun createFashionMNISTLoader(config: FashionMNISTLoaderConfig): FashionMNISTLoader {
    return FashionMNISTLoaderMacos(config)
}
