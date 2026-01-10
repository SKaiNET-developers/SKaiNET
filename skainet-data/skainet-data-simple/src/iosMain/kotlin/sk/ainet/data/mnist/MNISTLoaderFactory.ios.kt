package sk.ainet.data.mnist

public actual fun createMNISTLoader(config: MNISTLoaderConfig): MNISTLoader = MNISTLoaderIos(config)

public actual object MNISTLoaderFactory {
    public actual fun create(): MNISTLoader = MNISTLoaderIos.create()

    public actual fun create(cacheDir: String): MNISTLoader = MNISTLoaderIos.create(cacheDir)

    public actual fun create(config: MNISTLoaderConfig): MNISTLoader = MNISTLoaderIos.create(config)
}
