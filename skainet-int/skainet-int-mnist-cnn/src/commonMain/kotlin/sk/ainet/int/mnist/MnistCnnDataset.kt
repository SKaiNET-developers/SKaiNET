package sk.ainet.int.mnist

import sk.ainet.context.ExecutionContext
import sk.ainet.data.DataBatch
import sk.ainet.data.Dataset
import sk.ainet.data.mnist.MNIST
import sk.ainet.data.mnist.MNISTDataset
import sk.ainet.data.mnist.MNISTLoaderConfig
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8

/**
 * Dataset integration helpers for the MNIST-CNN Integration module.
 *
 * These APIs provide convenient access to the MNIST dataset using skainet-data-api/simple,
 * demonstrate batching via the Dataset API, and include helpers to convert memory-efficient
 * Int8 image tensors into FP32 normalized [0,1] tensors suitable for model inference.
 */
public object MnistCnnDataset {
    /** Load the MNIST train dataset using skainet-data-simple. */
    public suspend fun loadTrain(config: MNISTLoaderConfig = MNISTLoaderConfig()): MNISTDataset =
        MNIST.loadTrain(config)

    /** Load the MNIST test dataset using skainet-data-simple. */
    public suspend fun loadTest(config: MNISTLoaderConfig = MNISTLoaderConfig()): MNISTDataset =
        MNIST.loadTest(config)

    /**
     * Create an iterator that yields DataBatch instances over the given dataset.
     * The dataset implementation returns Int8 tensors for memory efficiency.
     */
    public fun batches(dataset: MNISTDataset, batchSize: Int): Iterator<DataBatch<Int8, Byte>> =
        dataset.batchIterator(batchSize)

    /**
     * Convert an Int8 image tensor [batch, 1, 28, 28] into FP32 normalized [0,1].
     *
     * The Int8 Byte values are expected to be unsigned pixel intensities 0..255 stored in Byte.
     * We convert using byte.toInt() and mask to 0..255, then divide by 255f.
     */
    public fun normalizeImagesToFp32(
        exec: ExecutionContext,
        int8Images: Tensor<Int8, Byte>,
        maxValue: Float = 255f
    ): Tensor<FP32, Float> {
        // Validate expected rank and channel dimension but do not over-constrain
        val shape = int8Images.shape // [B, 1, 28, 28]
        require(shape.rank == 4) { "Expected BCHW rank-4 tensor, got rank=${shape.rank} shape=$shape" }

        val volume = shape.volume
        val out = exec.zeros<FP32, Float>(shape, FP32::class)
        // Access through data buffers; generic API requires element-wise set/get
        // Iterate over all elements; optimize later if needed.
        var idx = 0
        for (b in 0 until shape[0]) {
            for (c in 0 until shape[1]) {
                for (h in 0 until shape[2]) {
                    for (w in 0 until shape[3]) {
                        val v: Int = int8Images.data[b, c, h, w].toInt() and 0xFF
                        val f = (v.toFloat() / maxValue)
                        out.data[b, c, h, w] = f
                        idx++
                    }
                }
            }
        }
        return out
    }

    /**
     * Convert a DataBatch<Int8, Byte> produced by MNISTDataset into a pair where
     * X is converted to FP32 normalized images and Y labels are returned as FP32 vector [batch].
     * This is helpful for model inference or loss evaluation flows that expect FP32.
     */
    public fun toFp32Batch(
        exec: ExecutionContext,
        batch: DataBatch<Int8, Byte>
    ): Pair<Array<Tensor<FP32, Float>>, Tensor<FP32, Float>> {
        require(batch.x.size == 1) { "MNIST expects single input tensor; got ${batch.x.size}" }
        val xInt8: Tensor<Int8, Byte> = batch.x[0]
        val xFp32 = normalizeImagesToFp32(exec, xInt8)

        val yShape = batch.y.shape // [batch]
        val yOut = exec.zeros<FP32, Float>(yShape, FP32::class)
        for (i in 0 until yShape[0]) {
            val labelInt: Int = batch.y.data[i].toInt() and 0xFF
            yOut.data[i] = labelInt.toFloat()
        }
        return arrayOf(xFp32) to yOut
    }
}
