package sk.ainet.data.mnist

public actual object MNISTLoaderFactory {
    public actual fun create(): MNISTLoader = MNISTLoaderLinux.create()

    public actual fun create(cacheDir: String): MNISTLoader = MNISTLoaderLinux.create(cacheDir)

    public actual fun create(config: MNISTLoaderConfig): MNISTLoader = MNISTLoaderLinux.create(config)
}