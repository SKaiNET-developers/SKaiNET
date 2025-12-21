package sk.ainet.apps.knanogpt.data

import sk.ainet.apps.knanogpt.CharTokenizer
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


class  ResourcesDataProvider<T: DType,V>(resourceName: String) : DataProvider<Tensor<T, V>> {

    private val textContent: String = ""
    override fun load(): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    init {
        /*

        val url = ResourcesDataProvider::class.java.classLoader.getResource(resourceName)
        val uri = url?.toURI() ?: throw IllegalArgumentException("File not found in resources.")

        val path = Paths.get(uri)
        textContent = String(Files.readAllBytes(path))

         */
    }

    /*
    override fun load(): Tensor<T,V> =
        Tensor(Shape(textContent.length), CharTokenizer(textContent).encode(textContent).map {
            it.toDouble()
        }.toDoubleArray())

     */
}