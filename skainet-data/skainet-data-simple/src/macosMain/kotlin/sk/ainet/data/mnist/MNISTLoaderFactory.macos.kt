package sk.ainet.data.mnist

public actual object MNISTLoaderFactory {
    public actual fun create(): MNISTLoader = MNISTLoaderMacos.create()

    public actual fun create(cacheDir: String): MNISTLoader = MNISTLoaderMacos.create(cacheDir)

    public actual fun create(config: MNISTLoaderConfig): MNISTLoader = MNISTLoaderMacos.create(config)
}