package sk.ainet.data.iris

/**
 * Entry point for the Iris dataset, mirroring the other SKaiNET data
 * providers (`MNIST`, `FashionMNIST`, `CIFAR10`).
 *
 * The dataset ships embedded inside the library, so [load] needs no network
 * access, no cache directory and works identically on every platform target.
 *
 * Example:
 * ```kotlin
 * val (train, test) = Iris.load().split(0.8, seed = 42L, stratified = true)
 * ```
 */
public object Iris {

    /**
     * Feature column order used by every tensor this provider produces.
     *
     * The order is part of the public contract: never rely on map iteration
     * order or CSV field position — index into feature arrays with these names.
     */
    public val featureNames: List<String> =
        listOf("sepalLength", "sepalWidth", "petalLength", "petalWidth")

    /**
     * Species names indexed by class label. The mapping is fixed and
     * alphabetical: 0 = "Iris-setosa", 1 = "Iris-versicolor",
     * 2 = "Iris-virginica". Stratified splits and one-hot batches both
     * depend on this ordering staying stable.
     */
    public val classNames: List<String> =
        listOf("Iris-setosa", "Iris-versicolor", "Iris-virginica")

    /**
     * Loads the bundled copy of the Iris dataset.
     *
     * The function is `suspend` for call-site symmetry with the downloading
     * providers (`MNIST.loadTrain()` & co.) even though loading is purely
     * in-memory parsing.
     */
    @Suppress("RedundantSuspendModifier")
    public suspend fun load(): IrisDataset = IrisDataset(parseIrisCsv(IRIS_CSV))
}
