# Java Model Training Guide

This guide covers building neural networks, defining loss functions and optimizers, loading datasets, and running training loops -- all from plain Java.

## Prerequisites

- JDK 21+ with `--enable-preview --add-modules jdk.incubator.vector`
- See [Java Getting Started](java-getting-started.md) for project setup

### Maven Dependencies

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>sk.ainet</groupId>
            <artifactId>skainet-bom</artifactId>
            <version>0.13.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<dependencies>
    <!-- Core tensor library + model builder -->
    <dependency>
        <groupId>sk.ainet</groupId>
        <artifactId>skainet-lang-core-jvm</artifactId>
    </dependency>

    <!-- CPU backend -->
    <dependency>
        <groupId>sk.ainet</groupId>
        <artifactId>skainet-backend-cpu-jvm</artifactId>
    </dependency>

    <!-- MNIST dataset loader -->
    <dependency>
        <groupId>sk.ainet</groupId>
        <artifactId>skainet-data-simple-jvm</artifactId>
    </dependency>
</dependencies>
```

---

## Building a Model with SequentialModelBuilder

`SequentialModelBuilder` provides a fluent API for stacking dense layers and activations. It lives in `sk.ainet.java`.

```java
import sk.ainet.java.SKaiNET;
import sk.ainet.java.SequentialModelBuilder;
import sk.ainet.lang.nn.Module;
import sk.ainet.lang.types.DType;

var ctx = SKaiNET.context();

Module model = new SequentialModelBuilder(ctx)
        .input(784)       // 28x28 flattened MNIST images
        .dense(128)       // fully connected: 784 -> 128
        .relu()           // ReLU activation
        .dense(10)        // fully connected: 128 -> 10 (digit classes)
        .build();
```

### Available Layers and Activations

| Method                  | Description                              |
|-------------------------|------------------------------------------|
| `.input(size)`          | Set the input dimension (must be first)  |
| `.dense(outputSize)`    | Fully connected (linear) layer           |
| `.relu()`               | ReLU activation: max(0, x)               |
| `.sigmoid()`            | Sigmoid activation                       |
| `.silu()`               | SiLU / Swish activation: x * sigmoid(x) |
| `.gelu()`               | GELU activation                          |
| `.softmax(dim)`         | Softmax along a dimension (default: -1)  |
| `.flatten(start, end)`  | Flatten dimensions                       |

Weights are initialized using Xavier initialization. The data type defaults to FP32; pass a `DType` to the constructor to change it:

```java
Module model = new SequentialModelBuilder(ctx, DType.fp16())
        .input(784)
        .dense(256)
        .gelu()
        .dense(10)
        .build();
```

---

## Losses

The `Losses` factory (in `sk.ainet.java`) creates loss function instances:

```java
import sk.ainet.java.Losses;
import sk.ainet.lang.nn.loss.Loss;

Loss ce   = Losses.crossEntropy();                  // log-softmax + NLL
Loss cce  = Losses.categoricalCrossEntropy();       // alias for cross-entropy
Loss scce = Losses.sparseCategoricalCrossEntropy();  // integer target indices
Loss mse  = Losses.mse();                            // mean squared error
Loss mae  = Losses.mae();                            // mean absolute error
Loss bce  = Losses.binaryCrossEntropy();             // binary cross-entropy
Loss bcl  = Losses.bceWithLogits();                  // BCE with logits (stable)
Loss hub  = Losses.huber(1.0f);                      // Huber / Smooth L1
Loss hin  = Losses.hinge(1.0f);                      // hinge loss
Loss shin = Losses.squaredHinge(1.0f);               // squared hinge
Loss poi  = Losses.poisson();                        // Poisson NLL
```

---

## Optimizers

The `Optimizers` factory (in `sk.ainet.java`) creates optimizer instances:

```java
import sk.ainet.java.Optimizers;
import sk.ainet.lang.nn.optim.Optimizer;

// Adam (default lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weightDecay=0.0)
Optimizer adam = Optimizers.adam(0.001);

// Adam with all parameters
Optimizer adamFull = Optimizers.adam(0.0003, 0.9, 0.999, 1e-8, 0.01);

// AdamW (decoupled weight decay, default weightDecay=0.01)
Optimizer adamw = Optimizers.adamw(0.001);

// SGD with momentum
Optimizer sgd = Optimizers.sgd(0.01, 0.9);

// SGD with momentum and weight decay
Optimizer sgdWd = Optimizers.sgd(0.01, 0.9, 0.0001);
```

---

## TrainingLoop

`TrainingLoop` ties together a model, loss function, optimizer, and execution context. Build it with the static builder:

```java
import sk.ainet.java.TrainingLoop;

TrainingLoop loop = TrainingLoop.builder()
        .model(model)
        .loss(Losses.crossEntropy())
        .optimizer(Optimizers.adam(0.001))
        .context(ctx)
        .build();
```

### Single Training Step

`step(x, y)` performs one forward pass, computes the loss, backpropagates, and updates weights. It returns the loss as a `float`:

```java
float loss = loop.step(inputBatch, targetBatch);
System.out.printf("Step loss: %.4f%n", loss);
```

### Full Training with `.train()`

`train()` accepts a `Supplier` that produces an `Iterator` of `(input, target)` pairs for each epoch:

```java
import sk.ainet.java.TrainingResult;
import kotlin.Pair;

TrainingResult result = loop.train(
        () -> batches.iterator(),  // called once per epoch
        10                          // number of epochs
);

System.out.printf("Trained %d epochs, final loss: %.4f%n",
        result.getEpochs(), result.getFinalLoss());
```

Each call to the supplier should return a fresh iterator over the training batches for that epoch. This allows reshuffling between epochs.

### Async Training with `.trainAsync()`

`trainAsync()` runs the training loop on a virtual thread and returns a `CompletableFuture<TrainingResult>`:

```java
import java.util.concurrent.CompletableFuture;

CompletableFuture<TrainingResult> future = loop.trainAsync(
        () -> batches.iterator(),
        20  // epochs
);

// Do other work while training runs in the background...

TrainingResult result = future.join();
System.out.printf("Final loss: %.4f%n", result.getFinalLoss());
```

You can also compose the future:

```java
loop.trainAsync(() -> batches.iterator(), 10)
    .thenAccept(r -> System.out.println("Done! Loss: " + r.getFinalLoss()))
    .exceptionally(ex -> { ex.printStackTrace(); return null; });
```

---

## Loading MNIST Data

The MNIST dataset loader lives in `sk.ainet.data.mnist`. The `MNISTBlocking` class provides blocking (non-suspend) methods for Java:

```java
import sk.ainet.data.mnist.MNISTBlocking;
import sk.ainet.data.mnist.MNISTDataset;

// Download and load training data (cached locally in ./mnist-data/)
MNISTDataset train = MNISTBlocking.loadTrain();
MNISTDataset test  = MNISTBlocking.loadTest();

System.out.println("Training samples: " + train.getImages().size());  // 60000
System.out.println("Test samples:     " + test.getImages().size());   // 10000
```

The first call downloads the dataset from the internet and caches it. Subsequent calls load from disk.

### Custom Cache Directory

```java
import sk.ainet.data.mnist.MNISTLoaderConfig;

MNISTLoaderConfig config = new MNISTLoaderConfig("/tmp/my-mnist-cache", true);
MNISTDataset train = MNISTBlocking.loadTrain(config);
```

### Working with MNIST Data

Each `MNISTDataset` contains a list of `MNISTImage` objects. Each image has a `byte[]` of 784 pixels (28x28) and a `byte` label (0-9):

```java
var firstImage = train.getImages().get(0);
byte label = firstImage.getLabel();       // e.g. 5
byte[] pixels = firstImage.getImage();    // 784 bytes, 0-255
```

### Creating Tensor Batches

To feed MNIST data into the training loop, convert images to tensors:

```java
import sk.ainet.java.SKaiNET;
import sk.ainet.lang.types.DType;
import kotlin.Pair;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

var ctx = SKaiNET.context();
int batchSize = 64;

List<Pair<Object, Object>> batches = new ArrayList<>();
var images = train.getImages();

for (int i = 0; i < images.size(); i += batchSize) {
    int end = Math.min(i + batchSize, images.size());
    int actual = end - i;

    // Flatten and normalize pixel data to [0, 1]
    float[] xData = new float[actual * 784];
    float[] yData = new float[actual];

    for (int j = 0; j < actual; j++) {
        var img = images.get(i + j);
        yData[j] = img.getLabel();
        byte[] px = img.getImage();
        for (int k = 0; k < 784; k++) {
            xData[j * 784 + k] = (px[k] & 0xFF) / 255.0f;
        }
    }

    var x = SKaiNET.tensor(ctx, new int[]{actual, 784}, DType.fp32(), xData);
    var y = SKaiNET.tensor(ctx, new int[]{actual}, DType.fp32(), yData);
    batches.add(new Pair<>(x, y));
}
```

---

## Complete MNIST Training Example

Putting it all together:

```java
package com.example;

import sk.ainet.java.*;
import sk.ainet.data.mnist.MNISTBlocking;
import sk.ainet.data.mnist.MNISTDataset;
import sk.ainet.lang.nn.Module;
import sk.ainet.lang.types.DType;
import kotlin.Pair;
import sk.ainet.lang.tensor.Tensor;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class MnistTraining {

    public static void main(String[] args) {
        // 1. Create context
        var ctx = SKaiNET.context();

        // 2. Build model: 784 -> 128 (ReLU) -> 10
        Module model = new SequentialModelBuilder(ctx)
                .input(784)
                .dense(128)
                .relu()
                .dense(10)
                .build();

        // 3. Set up training loop
        TrainingLoop loop = TrainingLoop.builder()
                .model(model)
                .loss(Losses.crossEntropy())
                .optimizer(Optimizers.adam(0.001))
                .context(ctx)
                .build();

        // 4. Load MNIST
        MNISTDataset train = MNISTBlocking.loadTrain();
        System.out.println("Loaded " + train.getImages().size() + " training samples");

        // 5. Create batches
        int batchSize = 64;
        List<Pair<Tensor, Tensor>> batches = createBatches(ctx, train, batchSize);

        // 6. Train for 5 epochs
        @SuppressWarnings("unchecked")
        TrainingResult result = loop.train(
                () -> (Iterator) batches.iterator(),
                5
        );

        System.out.printf("Training complete: %d epochs, final loss: %.4f%n",
                result.getEpochs(), result.getFinalLoss());
    }

    private static List<Pair<Tensor, Tensor>> createBatches(
            Object ctx, MNISTDataset dataset, int batchSize) {

        var context = (sk.ainet.context.ExecutionContext) ctx;
        var images = dataset.getImages();
        List<Pair<Tensor, Tensor>> batches = new ArrayList<>();

        for (int i = 0; i < images.size(); i += batchSize) {
            int end = Math.min(i + batchSize, images.size());
            int actual = end - i;

            float[] xData = new float[actual * 784];
            float[] yData = new float[actual];

            for (int j = 0; j < actual; j++) {
                var img = images.get(i + j);
                yData[j] = img.getLabel();
                byte[] px = img.getImage();
                for (int k = 0; k < 784; k++) {
                    xData[j * 784 + k] = (px[k] & 0xFF) / 255.0f;
                }
            }

            var x = SKaiNET.tensor(context, new int[]{actual, 784}, DType.fp32(), xData);
            var y = SKaiNET.tensor(context, new int[]{actual}, DType.fp32(), yData);
            batches.add(new Pair<>(x, y));
        }

        return batches;
    }
}
```

Run with:

```bash
java --enable-preview --add-modules jdk.incubator.vector \
     -cp target/classes:target/dependency/* \
     com.example.MnistTraining
```

---

## Async Training Example

For non-blocking training, use `trainAsync()` and handle the result with `CompletableFuture`:

```java
var future = loop.trainAsync(() -> (Iterator) batches.iterator(), 10);

// Monitor progress or do other work
System.out.println("Training started on virtual thread...");

future.thenAccept(result -> {
    System.out.printf("Finished: %d epochs, loss %.4f%n",
            result.getEpochs(), result.getFinalLoss());
}).join();
```

---

## Package Reference

| Package               | Key Classes                                          |
|-----------------------|------------------------------------------------------|
| `sk.ainet.java`      | `SKaiNET`, `SequentialModelBuilder`, `TrainingLoop`, `TrainingResult`, `Losses`, `Optimizers`, `TensorJavaOps` |
| `sk.ainet.data.mnist` | `MNISTBlocking`, `MNISTDataset`, `MNISTImage`, `MNISTLoaderConfig` |
| `sk.ainet.lang.types` | `DType`                                              |
| `sk.ainet.lang.nn.loss` | `Loss` (interface returned by `Losses` factory)    |
| `sk.ainet.lang.nn.optim` | `Optimizer` (interface returned by `Optimizers` factory) |

---

## Next Steps

- [Java Getting Started](java-getting-started.md) -- tensor operations, project setup, and dependency management.
- [LLM Inference Guide](java-llm-inference.md) -- load GGUF/SafeTensors models, generate text, and build agents.
