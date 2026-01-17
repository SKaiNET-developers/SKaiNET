package sk.ainet.exec.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.metrics.Accuracy
import sk.ainet.lang.nn.metrics.accuracy
import sk.ainet.lang.nn.metrics.binaryAccuracy
import sk.ainet.lang.nn.metrics.computeForBatch
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

class AccuracyTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    private fun intTensor(shape: Shape, data: IntArray): Tensor<Int32, Int> =
        ctx.fromIntArray(shape, Int32::class, data)

    @Test
    fun accuracy_with_index_targets_all_correct() {
        // Predictions: batch of 4, 3 classes
        // Shape: [4, 3]
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts class 2
                0.8f, 0.1f, 0.1f,  // predicts class 0
                0.1f, 0.8f, 0.1f,  // predicts class 1
                0.3f, 0.3f, 0.4f   // predicts class 2
            )
        )

        // Targets: class indices
        val targets = intTensor(Shape(4), intArrayOf(2, 0, 1, 2))

        val metric = accuracy()
        metric.update(preds, targets, ctx)

        assertEquals(1.0, metric.compute(), 1e-6)
    }

    @Test
    fun accuracy_with_index_targets_partial_correct() {
        // Predictions: batch of 4, 3 classes
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts class 2
                0.8f, 0.1f, 0.1f,  // predicts class 0
                0.1f, 0.8f, 0.1f,  // predicts class 1
                0.3f, 0.3f, 0.4f   // predicts class 2
            )
        )

        // Targets: 2 correct, 2 wrong
        val targets = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))  // class 2 correct, class 1 wrong, class 1 correct, class 2 wrong

        val metric = accuracy()
        metric.update(preds, targets, ctx)

        assertEquals(0.5, metric.compute(), 1e-6)
    }

    @Test
    fun accuracy_with_soft_targets() {
        // Predictions: batch of 3, 3 classes
        val preds = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts class 2
                0.8f, 0.1f, 0.1f,  // predicts class 0
                0.1f, 0.8f, 0.1f   // predicts class 1
            )
        )

        // One-hot encoded targets
        val targets = tensor(
            Shape(3, 3),
            floatArrayOf(
                0f, 0f, 1f,  // class 2
                1f, 0f, 0f,  // class 0
                0f, 1f, 0f   // class 1
            )
        )

        val metric = accuracy()
        metric.update(preds, targets, ctx)

        assertEquals(1.0, metric.compute(), 1e-6)
    }

    @Test
    fun accuracy_accumulates_over_batches() {
        val metric = accuracy()

        // First batch: 2 correct out of 2
        val preds1 = tensor(Shape(2, 2), floatArrayOf(0.9f, 0.1f, 0.1f, 0.9f))
        val targets1 = intTensor(Shape(2), intArrayOf(0, 1))
        metric.update(preds1, targets1, ctx)

        assertEquals(1.0, metric.compute(), 1e-6)

        // Second batch: 1 correct out of 2
        val preds2 = tensor(Shape(2, 2), floatArrayOf(0.9f, 0.1f, 0.9f, 0.1f))  // predicts 0, 0
        val targets2 = intTensor(Shape(2), intArrayOf(0, 1))  // expects 0, 1
        metric.update(preds2, targets2, ctx)

        // Total: 3 correct out of 4
        assertEquals(0.75, metric.compute(), 1e-6)
    }

    @Test
    fun accuracy_reset_clears_state() {
        val metric = accuracy()

        // Add some data
        val preds = tensor(Shape(2, 2), floatArrayOf(0.9f, 0.1f, 0.1f, 0.9f))
        val targets = intTensor(Shape(2), intArrayOf(0, 1))
        metric.update(preds, targets, ctx)

        assertEquals(1.0, metric.compute(), 1e-6)

        // Reset
        metric.reset()

        // Should be 0 after reset (no data)
        assertEquals(0.0, metric.compute(), 1e-6)
    }

    @Test
    fun accuracy_compute_for_batch_helper() {
        val preds = tensor(Shape(2, 2), floatArrayOf(0.9f, 0.1f, 0.1f, 0.9f))
        val targets = intTensor(Shape(2), intArrayOf(0, 1))

        val metric = accuracy()
        val result = metric.computeForBatch(preds, targets, ctx)

        assertEquals(1.0, result, 1e-6)
    }

    @Test
    fun binary_accuracy_with_threshold() {
        // Binary classification with sigmoid-like outputs
        val preds = tensor(
            Shape(4, 1),
            floatArrayOf(
                0.3f,  // < 0.5, predicts 0
                0.7f,  // > 0.5, predicts 1
                0.6f,  // > 0.5, predicts 1
                0.4f   // < 0.5, predicts 0
            )
        )

        val targets = intTensor(Shape(4), intArrayOf(0, 1, 1, 0))

        val metric = binaryAccuracy()
        metric.update(preds, targets, ctx)

        assertEquals(1.0, metric.compute(), 1e-6)
    }

    @Test
    fun binary_accuracy_with_custom_threshold() {
        val preds = tensor(Shape(4, 1), floatArrayOf(0.6f, 0.8f, 0.65f, 0.5f))

        // With threshold 0.7: predictions are 0, 1, 0, 0
        val targets = intTensor(Shape(4), intArrayOf(0, 1, 0, 0))

        val metric = Accuracy(threshold = 0.7f)
        metric.update(preds, targets, ctx)

        assertEquals(1.0, metric.compute(), 1e-6)
    }

    @Test
    fun accuracy_with_batch_dimension() {
        // Shape: [batch=2, seq=3, classes=4]
        val preds = tensor(
            Shape(2, 3, 4),
            floatArrayOf(
                // Batch 0
                0.1f, 0.2f, 0.3f, 0.4f,  // seq 0: predicts class 3
                0.4f, 0.3f, 0.2f, 0.1f,  // seq 1: predicts class 0
                0.1f, 0.4f, 0.3f, 0.2f,  // seq 2: predicts class 1
                // Batch 1
                0.1f, 0.1f, 0.7f, 0.1f,  // seq 0: predicts class 2
                0.4f, 0.3f, 0.2f, 0.1f,  // seq 1: predicts class 0
                0.1f, 0.1f, 0.1f, 0.7f   // seq 2: predicts class 3
            )
        )

        // Targets shape: [batch=2, seq=3]
        val targets = intTensor(
            Shape(2, 3),
            intArrayOf(
                3, 0, 1,  // Batch 0: all correct
                2, 0, 3   // Batch 1: all correct
            )
        )

        val metric = accuracy(dim = -1)  // class dimension is last
        metric.update(preds, targets, ctx)

        assertEquals(1.0, metric.compute(), 1e-6)
    }

    @Test
    fun accuracy_empty_returns_zero() {
        val metric = accuracy()
        assertEquals(0.0, metric.compute(), 1e-6)
    }

    @Test
    fun accuracy_name_property() {
        val metric = accuracy()
        assertEquals("accuracy", metric.name)
    }

    @Test
    fun accuracy_all_wrong() {
        val preds = tensor(
            Shape(3, 2),
            floatArrayOf(
                0.9f, 0.1f,  // predicts 0
                0.9f, 0.1f,  // predicts 0
                0.9f, 0.1f   // predicts 0
            )
        )

        val targets = intTensor(Shape(3), intArrayOf(1, 1, 1))

        val metric = accuracy()
        metric.update(preds, targets, ctx)

        assertEquals(0.0, metric.compute(), 1e-6)
    }
}
