package sk.ainet.data.iris

/**
 * One Iris flower: four physical measurements in centimetres plus its species index.
 *
 * @property sepalLength Sepal length in cm.
 * @property sepalWidth  Sepal width in cm.
 * @property petalLength Petal length in cm.
 * @property petalWidth  Petal width in cm.
 * @property label       Species as a class index into [Iris.classNames]
 *                       (0 = Iris-setosa, 1 = Iris-versicolor, 2 = Iris-virginica).
 */
public data class IrisSample(
    val sepalLength: Float,
    val sepalWidth: Float,
    val petalLength: Float,
    val petalWidth: Float,
    val label: Int
) {
    /** Returns the four measurements in the fixed feature order given by [Iris.featureNames]. */
    public fun toFeatures(): FloatArray =
        floatArrayOf(sepalLength, sepalWidth, petalLength, petalWidth)
}

/**
 * The complete Iris dataset (Fisher, 1936) embedded verbatim: 150 rows of
 * `sepalLength,sepalWidth,petalLength,petalWidth,species` in the canonical
 * UCI ordering (50 rows per species). The data is in the public domain.
 *
 * It is embedded as source rather than as a resource file so that every
 * platform target of this module — including JS and Wasm browsers — can load
 * it with no I/O and no network access.
 */
internal val IRIS_CSV: String = """
    5.1,3.5,1.4,0.2,Iris-setosa
    4.9,3.0,1.4,0.2,Iris-setosa
    4.7,3.2,1.3,0.2,Iris-setosa
    4.6,3.1,1.5,0.2,Iris-setosa
    5.0,3.6,1.4,0.2,Iris-setosa
    5.4,3.9,1.7,0.4,Iris-setosa
    4.6,3.4,1.4,0.3,Iris-setosa
    5.0,3.4,1.5,0.2,Iris-setosa
    4.4,2.9,1.4,0.2,Iris-setosa
    4.9,3.1,1.5,0.1,Iris-setosa
    5.4,3.7,1.5,0.2,Iris-setosa
    4.8,3.4,1.6,0.2,Iris-setosa
    4.8,3.0,1.4,0.1,Iris-setosa
    4.3,3.0,1.1,0.1,Iris-setosa
    5.8,4.0,1.2,0.2,Iris-setosa
    5.7,4.4,1.5,0.4,Iris-setosa
    5.4,3.9,1.3,0.4,Iris-setosa
    5.1,3.5,1.4,0.3,Iris-setosa
    5.7,3.8,1.7,0.3,Iris-setosa
    5.1,3.8,1.5,0.3,Iris-setosa
    5.4,3.4,1.7,0.2,Iris-setosa
    5.1,3.7,1.5,0.4,Iris-setosa
    4.6,3.6,1.0,0.2,Iris-setosa
    5.1,3.3,1.7,0.5,Iris-setosa
    4.8,3.4,1.9,0.2,Iris-setosa
    5.0,3.0,1.6,0.2,Iris-setosa
    5.0,3.4,1.6,0.4,Iris-setosa
    5.2,3.5,1.5,0.2,Iris-setosa
    5.2,3.4,1.4,0.2,Iris-setosa
    4.7,3.2,1.6,0.2,Iris-setosa
    4.8,3.1,1.6,0.2,Iris-setosa
    5.4,3.4,1.5,0.4,Iris-setosa
    5.2,4.1,1.5,0.1,Iris-setosa
    5.5,4.2,1.4,0.2,Iris-setosa
    4.9,3.1,1.5,0.1,Iris-setosa
    5.0,3.2,1.2,0.2,Iris-setosa
    5.5,3.5,1.3,0.2,Iris-setosa
    4.9,3.1,1.5,0.1,Iris-setosa
    4.4,3.0,1.3,0.2,Iris-setosa
    5.1,3.4,1.5,0.2,Iris-setosa
    5.0,3.5,1.3,0.3,Iris-setosa
    4.5,2.3,1.3,0.3,Iris-setosa
    4.4,3.2,1.3,0.2,Iris-setosa
    5.0,3.5,1.6,0.6,Iris-setosa
    5.1,3.8,1.9,0.4,Iris-setosa
    4.8,3.0,1.4,0.3,Iris-setosa
    5.1,3.8,1.6,0.2,Iris-setosa
    4.6,3.2,1.4,0.2,Iris-setosa
    5.3,3.7,1.5,0.2,Iris-setosa
    5.0,3.3,1.4,0.2,Iris-setosa
    7.0,3.2,4.7,1.4,Iris-versicolor
    6.4,3.2,4.5,1.5,Iris-versicolor
    6.9,3.1,4.9,1.5,Iris-versicolor
    5.5,2.3,4.0,1.3,Iris-versicolor
    6.5,2.8,4.6,1.5,Iris-versicolor
    5.7,2.8,4.5,1.3,Iris-versicolor
    6.3,3.3,4.7,1.6,Iris-versicolor
    4.9,2.4,3.3,1.0,Iris-versicolor
    6.6,2.9,4.6,1.3,Iris-versicolor
    5.2,2.7,3.9,1.4,Iris-versicolor
    5.0,2.0,3.5,1.0,Iris-versicolor
    5.9,3.0,4.2,1.5,Iris-versicolor
    6.0,2.2,4.0,1.0,Iris-versicolor
    6.1,2.9,4.7,1.4,Iris-versicolor
    5.6,2.9,3.6,1.3,Iris-versicolor
    6.7,3.1,4.4,1.4,Iris-versicolor
    5.6,3.0,4.5,1.5,Iris-versicolor
    5.8,2.7,4.1,1.0,Iris-versicolor
    6.2,2.2,4.5,1.5,Iris-versicolor
    5.6,2.5,3.9,1.1,Iris-versicolor
    5.9,3.2,4.8,1.8,Iris-versicolor
    6.1,2.8,4.0,1.3,Iris-versicolor
    6.3,2.5,4.9,1.5,Iris-versicolor
    6.1,2.8,4.7,1.2,Iris-versicolor
    6.4,2.9,4.3,1.3,Iris-versicolor
    6.6,3.0,4.4,1.4,Iris-versicolor
    6.8,2.8,4.8,1.4,Iris-versicolor
    6.7,3.0,5.0,1.7,Iris-versicolor
    6.0,2.9,4.5,1.5,Iris-versicolor
    5.7,2.6,3.5,1.0,Iris-versicolor
    5.5,2.4,3.8,1.1,Iris-versicolor
    5.5,2.4,3.7,1.0,Iris-versicolor
    5.8,2.7,3.9,1.2,Iris-versicolor
    6.0,2.7,5.1,1.6,Iris-versicolor
    5.4,3.0,4.5,1.5,Iris-versicolor
    6.0,3.4,4.5,1.6,Iris-versicolor
    6.7,3.1,4.7,1.5,Iris-versicolor
    6.3,2.3,4.4,1.3,Iris-versicolor
    5.6,3.0,4.1,1.3,Iris-versicolor
    5.5,2.5,4.0,1.3,Iris-versicolor
    5.5,2.6,4.4,1.2,Iris-versicolor
    6.1,3.0,4.6,1.4,Iris-versicolor
    5.8,2.6,4.0,1.2,Iris-versicolor
    5.0,2.3,3.3,1.0,Iris-versicolor
    5.6,2.7,4.2,1.3,Iris-versicolor
    5.7,3.0,4.2,1.2,Iris-versicolor
    5.7,2.9,4.2,1.3,Iris-versicolor
    6.2,2.9,4.3,1.3,Iris-versicolor
    5.1,2.5,3.0,1.1,Iris-versicolor
    5.7,2.8,4.1,1.3,Iris-versicolor
    6.3,3.3,6.0,2.5,Iris-virginica
    5.8,2.7,5.1,1.9,Iris-virginica
    7.1,3.0,5.9,2.1,Iris-virginica
    6.3,2.9,5.6,1.8,Iris-virginica
    6.5,3.0,5.8,2.2,Iris-virginica
    7.6,3.0,6.6,2.1,Iris-virginica
    4.9,2.5,4.5,1.7,Iris-virginica
    7.3,2.9,6.3,1.8,Iris-virginica
    6.7,2.5,5.8,1.8,Iris-virginica
    7.2,3.6,6.1,2.5,Iris-virginica
    6.5,3.2,5.1,2.0,Iris-virginica
    6.4,2.7,5.3,1.9,Iris-virginica
    6.8,3.0,5.5,2.1,Iris-virginica
    5.7,2.5,5.0,2.0,Iris-virginica
    5.8,2.8,5.1,2.4,Iris-virginica
    6.4,3.2,5.3,2.3,Iris-virginica
    6.5,3.0,5.5,1.8,Iris-virginica
    7.7,3.8,6.7,2.2,Iris-virginica
    7.7,2.6,6.9,2.3,Iris-virginica
    6.0,2.2,5.0,1.5,Iris-virginica
    6.9,3.2,5.7,2.3,Iris-virginica
    5.6,2.8,4.9,2.0,Iris-virginica
    7.7,2.8,6.7,2.0,Iris-virginica
    6.3,2.7,4.9,1.8,Iris-virginica
    6.7,3.3,5.7,2.1,Iris-virginica
    7.2,3.2,6.0,1.8,Iris-virginica
    6.2,2.8,4.8,1.8,Iris-virginica
    6.1,3.0,4.9,1.8,Iris-virginica
    6.4,2.8,5.6,2.1,Iris-virginica
    7.2,3.0,5.8,1.6,Iris-virginica
    7.4,2.8,6.1,1.9,Iris-virginica
    7.9,3.8,6.4,2.0,Iris-virginica
    6.4,2.8,5.6,2.2,Iris-virginica
    6.3,2.8,5.1,1.5,Iris-virginica
    6.1,2.6,5.6,1.4,Iris-virginica
    7.7,3.0,6.1,2.3,Iris-virginica
    6.3,3.4,5.6,2.4,Iris-virginica
    6.4,3.1,5.5,1.8,Iris-virginica
    6.0,3.0,4.8,1.8,Iris-virginica
    6.9,3.1,5.4,2.1,Iris-virginica
    6.7,3.1,5.6,2.4,Iris-virginica
    6.9,3.1,5.1,2.3,Iris-virginica
    5.8,2.7,5.1,1.9,Iris-virginica
    6.8,3.2,5.9,2.3,Iris-virginica
    6.7,3.3,5.7,2.5,Iris-virginica
    6.7,3.0,5.2,2.3,Iris-virginica
    6.3,2.5,5.0,1.9,Iris-virginica
    6.5,3.0,5.2,2.0,Iris-virginica
    6.2,3.4,5.4,2.3,Iris-virginica
    5.9,3.0,5.1,1.8,Iris-virginica
""".trimIndent()

/**
 * Parses the embedded Iris CSV into samples.
 *
 * Errors always name the offending line number and column, e.g. an unknown
 * species or a non-numeric measurement fails fast instead of silently
 * producing a broken sample.
 */
internal fun parseIrisCsv(csv: String): List<IrisSample> {
    val featureCount = Iris.featureNames.size
    return csv.lines()
        .withIndex()
        .filter { (_, line) -> line.isNotBlank() }
        .map { (index, line) ->
            val lineNumber = index + 1
            val fields = line.split(",")
            require(fields.size == featureCount + 1) {
                "Iris CSV line $lineNumber has ${fields.size} fields but expected ${featureCount + 1}"
            }

            fun featureAt(column: Int): Float =
                fields[column].trim().toFloatOrNull()
                    ?: throw IllegalArgumentException(
                        "Iris CSV line $lineNumber column '${Iris.featureNames[column]}' is not a valid number: '${fields[column].trim()}'"
                    )

            val speciesField = fields[featureCount].trim()
            val label = Iris.classNames.indexOf(speciesField)
            require(label >= 0) {
                "Iris CSV line $lineNumber has unknown species '$speciesField'; expected one of ${Iris.classNames}"
            }

            IrisSample(
                sepalLength = featureAt(0),
                sepalWidth = featureAt(1),
                petalLength = featureAt(2),
                petalWidth = featureAt(3),
                label = label
            )
        }
}
