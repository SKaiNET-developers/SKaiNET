package sk.ainet.examples;

import sk.ainet.java.*;
import sk.ainet.lang.types.DType;

/**
 * MNIST training example using the Java API.
 *
 * Demonstrates building a simple neural network, loading MNIST data,
 * and training with the TrainingLoop builder.
 */
public class MNISTTraining {
    public static void main(String[] args) {
        // 1. Create context
        var ctx = SKaiNET.context();

        // 2. Build model: 784 -> 128 (ReLU) -> 10
        var model = new SequentialModelBuilder(ctx)
                .input(784)
                .dense(128)
                .relu()
                .dense(64)
                .relu()
                .dense(10)
                .build();

        // 3. Create training loop
        var loop = TrainingLoop.builder()
                .model(model)
                .loss(Losses.crossEntropy())
                .optimizer(Optimizers.adam(0.001))
                .context(ctx)
                .build();

        System.out.println("Model and training loop created successfully.");
        System.out.println("To train on actual MNIST data, use MNISTBlocking.loadTrain().");

        // 4. Example single step with synthetic data
        var x = SKaiNET.randn(ctx, new int[]{32, 784});  // batch of 32
        var y = SKaiNET.zeros(ctx, new int[]{32}, DType.int32());  // labels

        float loss = loop.step(x, y);
        System.out.println("Training step loss: " + loss);
    }
}
