package sk.ainet.examples;

import sk.ainet.java.SKaiNET;
import sk.ainet.java.TensorJavaOps;
import sk.ainet.lang.types.DType;

/**
 * Hello Tensor — minimal SKaiNET Java example.
 *
 * Demonstrates creating tensors, performing matrix multiplication,
 * and applying activation functions using the Java API.
 *
 * Run with: mvn exec:java -Dexec.mainClass="sk.ainet.examples.HelloTensor"
 * JVM flags: --enable-preview --add-modules jdk.incubator.vector
 */
public class HelloTensor {
    public static void main(String[] args) {
        // 1. Create an execution context (CPU backend)
        var ctx = SKaiNET.context();

        // 2. Create tensors
        var a = SKaiNET.tensor(ctx, new int[]{2, 3}, DType.fp32(),
                new float[]{1, 2, 3, 4, 5, 6});
        var b = SKaiNET.tensor(ctx, new int[]{3, 2}, DType.fp32(),
                new float[]{7, 8, 9, 10, 11, 12});

        // 3. Matrix multiplication: [2,3] x [3,2] = [2,2]
        var c = TensorJavaOps.matmul(a, b);
        System.out.println("matmul result shape: " + java.util.Arrays.toString(c.getShape().getDimensions()));

        // 4. Apply ReLU activation
        var d = TensorJavaOps.relu(c);
        System.out.println("After ReLU — shape: " + java.util.Arrays.toString(d.getShape().getDimensions()));

        // 5. Create convenience tensors
        var zeros = SKaiNET.zeros(ctx, new int[]{3, 3});
        var ones = SKaiNET.ones(ctx, new int[]{3, 3});
        var sum = TensorJavaOps.add(zeros, ones);

        System.out.println("zeros + ones shape: " + java.util.Arrays.toString(sum.getShape().getDimensions()));
        System.out.println("Done!");
    }
}
