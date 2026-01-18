package sk.ainet.data.fashionmnist

/**
 * iOS implementation of the Fashion-MNIST loader factory.
 */
public actual fun createFashionMNISTLoader(config: FashionMNISTLoaderConfig): FashionMNISTLoader {
    return FashionMNISTLoaderIos(config)
}
