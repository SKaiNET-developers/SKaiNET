package sk.ainet.apps.knanogpt.data

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.random.Random

class BatchProvider(
    private val trainingData: Tensor<FP32,Float>,
    private val validationData: Tensor<FP32,Float>,
    private val blockSize: Int,
    private val batchSize: Int,
    private val random: Random = Random(1337)
) {

    /*
    fun getRandomBatch(selectTrainingData: Boolean): Pair<Tensor<>, Tensor> {
        // generate a small batch of data of inputs x and targets y
        val data = if (selectTrainingData) trainingData else validationData
        val ix = randint(random, 0, data.elements.size - blockSize, Shape(batchSize))
        val x = stack(ix.elements.map {
            Tensor(Shape(blockSize), data.elements.slice(it.toInt()..<it.toInt() + blockSize).toDoubleArray())
        })
        val y = stack(ix.elements.map {
            Tensor(Shape(blockSize), data.elements.slice(it.toInt() + 1..<it.toInt() + blockSize + 1).toDoubleArray())
        })
        return Pair(x, y)
    }


    fun getBatch(selectTrainingData: Boolean) {
        val data = if (selectTrainingData) trainingData else validationData
        val x = data[0..blockSize]
        val y = data[1..blockSize + 1]
        for (i in 0..<blockSize) { // `end` is included
            val context = x[0..<i + 1]
            val target = y[i]
            println("when input is $context the target: $target")
        }
    }

     */

}