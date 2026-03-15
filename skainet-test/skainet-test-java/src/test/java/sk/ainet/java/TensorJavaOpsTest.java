package sk.ainet.java;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import sk.ainet.context.ExecutionContext;
import sk.ainet.lang.tensor.Tensor;
import sk.ainet.lang.types.DType;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the Java-friendly tensor operations exposed by {@link TensorJavaOps}.
 */
class TensorJavaOpsTest {

    private static ExecutionContext ctx;

    @BeforeAll
    static void setUp() {
        ctx = SKaiNET.context();
    }

    // ---- Arithmetic ----

    @Test
    void add() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{2, 2}, DType.fp32(),
                new float[]{1f, 2f, 3f, 4f});
        Tensor<?, ?> b = SKaiNET.tensor(ctx, new int[]{2, 2}, DType.fp32(),
                new float[]{10f, 20f, 30f, 40f});

        Tensor<?, ?> c = TensorJavaOps.add(a, b);
        assertNotNull(c);
        assertArrayEquals(new int[]{2, 2}, c.getShape().getDimensions());

        float[] result = c.getData().copyToFloatArray();
        assertArrayEquals(new float[]{11f, 22f, 33f, 44f}, result, 1e-6f);
    }

    @Test
    void subtract() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{3}, DType.fp32(),
                new float[]{10f, 20f, 30f});
        Tensor<?, ?> b = SKaiNET.tensor(ctx, new int[]{3}, DType.fp32(),
                new float[]{1f, 2f, 3f});

        Tensor<?, ?> c = TensorJavaOps.subtract(a, b);
        float[] result = c.getData().copyToFloatArray();
        assertArrayEquals(new float[]{9f, 18f, 27f}, result, 1e-6f);
    }

    @Test
    void multiply() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{2, 2}, DType.fp32(),
                new float[]{1f, 2f, 3f, 4f});
        Tensor<?, ?> b = SKaiNET.tensor(ctx, new int[]{2, 2}, DType.fp32(),
                new float[]{5f, 6f, 7f, 8f});

        Tensor<?, ?> c = TensorJavaOps.multiply(a, b);
        float[] result = c.getData().copyToFloatArray();
        assertArrayEquals(new float[]{5f, 12f, 21f, 32f}, result, 1e-6f);
    }

    @Test
    void divide() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{4}, DType.fp32(),
                new float[]{10f, 20f, 30f, 40f});
        Tensor<?, ?> b = SKaiNET.tensor(ctx, new int[]{4}, DType.fp32(),
                new float[]{2f, 4f, 5f, 8f});

        Tensor<?, ?> c = TensorJavaOps.divide(a, b);
        float[] result = c.getData().copyToFloatArray();
        assertArrayEquals(new float[]{5f, 5f, 6f, 5f}, result, 1e-6f);
    }

    // ---- Linear Algebra ----

    @Test
    void matmul() {
        // [2x3] x [3x2] -> [2x2]
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{2, 3}, DType.fp32(),
                new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        Tensor<?, ?> b = SKaiNET.tensor(ctx, new int[]{3, 2}, DType.fp32(),
                new float[]{7f, 8f, 9f, 10f, 11f, 12f});

        Tensor<?, ?> c = TensorJavaOps.matmul(a, b);
        assertArrayEquals(new int[]{2, 2}, c.getShape().getDimensions());

        float[] result = c.getData().copyToFloatArray();
        // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assertArrayEquals(new float[]{58f, 64f, 139f, 154f}, result, 1e-4f);
    }

    // ---- Activation Functions ----

    @Test
    void relu() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{4}, DType.fp32(),
                new float[]{-2f, -1f, 0f, 3f});

        Tensor<?, ?> r = TensorJavaOps.relu(a);
        float[] result = r.getData().copyToFloatArray();
        assertArrayEquals(new float[]{0f, 0f, 0f, 3f}, result, 1e-6f);
    }

    @Test
    void sigmoid() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{3}, DType.fp32(),
                new float[]{0f, 100f, -100f});

        Tensor<?, ?> s = TensorJavaOps.sigmoid(a);
        float[] result = s.getData().copyToFloatArray();
        assertEquals(0.5f, result[0], 1e-5f, "sigmoid(0) should be 0.5");
        assertTrue(result[1] > 0.99f, "sigmoid(100) should be close to 1.0");
        assertTrue(result[2] < 0.01f, "sigmoid(-100) should be close to 0.0");
    }

    @Test
    void softmax() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{1, 4}, DType.fp32(),
                new float[]{1f, 2f, 3f, 4f});

        Tensor<?, ?> s = TensorJavaOps.softmax(a, -1);
        float[] result = s.getData().copyToFloatArray();

        // All values should be positive
        for (float v : result) {
            assertTrue(v > 0f, "softmax values should be positive");
        }

        // Values should sum to approximately 1.0
        float sum = 0f;
        for (float v : result) {
            sum += v;
        }
        assertEquals(1.0f, sum, 1e-5f, "softmax values should sum to 1.0");

        // Values should be in ascending order (since input is ascending)
        for (int i = 1; i < result.length; i++) {
            assertTrue(result[i] > result[i - 1],
                    "softmax should preserve ordering of ascending inputs");
        }
    }

    // ---- Shape Operations ----

    @Test
    void reshape() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{2, 3}, DType.fp32(),
                new float[]{1f, 2f, 3f, 4f, 5f, 6f});

        Tensor<?, ?> r = TensorJavaOps.reshape(a, new int[]{3, 2});
        assertArrayEquals(new int[]{3, 2}, r.getShape().getDimensions());
        assertEquals(6, r.getVolume());
    }

    @Test
    void flatten() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{2, 3}, DType.fp32(),
                new float[]{1f, 2f, 3f, 4f, 5f, 6f});

        Tensor<?, ?> f = TensorJavaOps.flatten(a, 0, -1);
        assertEquals(1, f.getRank(), "Fully flattened tensor should have rank 1");
        assertEquals(6, f.getVolume());
    }

    @Test
    void squeeze() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{1, 3, 1}, DType.fp32(),
                new float[]{1f, 2f, 3f});

        Tensor<?, ?> s = TensorJavaOps.squeeze(a, null);
        assertArrayEquals(new int[]{3}, s.getShape().getDimensions());
    }

    @Test
    void unsqueeze() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{3}, DType.fp32(),
                new float[]{1f, 2f, 3f});

        Tensor<?, ?> u = TensorJavaOps.unsqueeze(a, 0);
        assertArrayEquals(new int[]{1, 3}, u.getShape().getDimensions());
    }

    // ---- Reductions ----

    @Test
    void sum() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{2, 3}, DType.fp32(),
                new float[]{1f, 2f, 3f, 4f, 5f, 6f});

        Tensor<?, ?> s = TensorJavaOps.sum(a, null);
        float[] result = s.getData().copyToFloatArray();
        assertEquals(21f, result[0], 1e-5f, "Sum of 1..6 should be 21");
    }

    @Test
    void mean() {
        Tensor<?, ?> a = SKaiNET.tensor(ctx, new int[]{4}, DType.fp32(),
                new float[]{2f, 4f, 6f, 8f});

        Tensor<?, ?> m = TensorJavaOps.mean(a, null);
        float[] result = m.getData().copyToFloatArray();
        assertEquals(5f, result[0], 1e-5f, "Mean of {2,4,6,8} should be 5");
    }
}
