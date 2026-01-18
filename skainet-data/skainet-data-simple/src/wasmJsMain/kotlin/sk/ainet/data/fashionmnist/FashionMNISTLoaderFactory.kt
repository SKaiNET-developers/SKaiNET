package sk.ainet.data.fashionmnist

/**
 * WASM JS implementation of the Fashion-MNIST loader factory.
 */
public actual fun createFashionMNISTLoader(config: FashionMNISTLoaderConfig): FashionMNISTLoader {
    return FashionMNISTLoaderWasmJs(config)
}
