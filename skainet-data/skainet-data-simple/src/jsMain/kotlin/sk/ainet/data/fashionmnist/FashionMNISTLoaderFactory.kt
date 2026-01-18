package sk.ainet.data.fashionmnist

/**
 * JS implementation of the Fashion-MNIST loader factory.
 */
public actual fun createFashionMNISTLoader(config: FashionMNISTLoaderConfig): FashionMNISTLoader {
    return FashionMNISTLoaderJs(config)
}
