package sk.ainet.data.cifar10

/**
 * JVM implementation of the CIFAR-10 loader factory.
 */
public actual fun createCIFAR10Loader(config: CIFAR10LoaderConfig): CIFAR10Loader {
    return CIFAR10LoaderJvm(config)
}
