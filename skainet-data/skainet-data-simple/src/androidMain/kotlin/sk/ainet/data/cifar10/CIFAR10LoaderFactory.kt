package sk.ainet.data.cifar10

/**
 * Android implementation of the CIFAR-10 loader factory.
 */
public actual fun createCIFAR10Loader(config: CIFAR10LoaderConfig): CIFAR10Loader {
    return CIFAR10LoaderAndroid(config)
}
