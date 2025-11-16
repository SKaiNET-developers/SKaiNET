package sk.ainet.data.mnist

public actual fun createMNISTLoader(config: MNISTLoaderConfig): MNISTLoader = MNISTLoaderJvm(config)
