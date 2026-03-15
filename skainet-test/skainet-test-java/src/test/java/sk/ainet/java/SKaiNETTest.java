package sk.ainet.java;

import org.junit.jupiter.api.Test;
import sk.ainet.context.ExecutionContext;
import sk.ainet.lang.tensor.Tensor;
import sk.ainet.lang.types.DType;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the Java entry-point API surface exposed by {@link SKaiNET}.
 */
class SKaiNETTest {

    @Test
    void contextCreation() {
        ExecutionContext ctx = SKaiNET.context();
        assertNotNull(ctx, "context() must return a non-null ExecutionContext");
    }

    @Test
    void tensorFromFloatArray() {
        ExecutionContext ctx = SKaiNET.context();
        float[] data = {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor<?, ?> t = SKaiNET.tensor(ctx, new int[]{2, 3}, DType.fp32(), data);

        assertNotNull(t, "tensor() must return a non-null Tensor");
        assertArrayEquals(new int[]{2, 3}, t.getShape().getDimensions());
        assertEquals(6, t.getVolume());
    }

    @Test
    void zerosDefaultDtype() {
        ExecutionContext ctx = SKaiNET.context();
        Tensor<?, ?> z = SKaiNET.zeros(ctx, new int[]{3, 4});

        assertNotNull(z);
        assertArrayEquals(new int[]{3, 4}, z.getShape().getDimensions());
        assertEquals(12, z.getVolume());
    }

    @Test
    void zerosExplicitDtype() {
        ExecutionContext ctx = SKaiNET.context();
        Tensor<?, ?> z = SKaiNET.zeros(ctx, new int[]{2, 2}, DType.fp32());

        assertNotNull(z);
        assertArrayEquals(new int[]{2, 2}, z.getShape().getDimensions());
    }

    @Test
    void onesDefaultDtype() {
        ExecutionContext ctx = SKaiNET.context();
        Tensor<?, ?> o = SKaiNET.ones(ctx, new int[]{4});

        assertNotNull(o);
        assertArrayEquals(new int[]{4}, o.getShape().getDimensions());
    }

    @Test
    void onesExplicitDtype() {
        ExecutionContext ctx = SKaiNET.context();
        Tensor<?, ?> o = SKaiNET.ones(ctx, new int[]{2, 3}, DType.fp32());

        assertNotNull(o);
        assertEquals(6, o.getVolume());
    }

    @Test
    void fullTensor() {
        ExecutionContext ctx = SKaiNET.context();
        Tensor<?, ?> f = SKaiNET.full(ctx, new int[]{2, 2}, DType.fp32(), 7.0f);

        assertNotNull(f);
        assertArrayEquals(new int[]{2, 2}, f.getShape().getDimensions());

        float[] values = f.getData().copyToFloatArray();
        for (float v : values) {
            assertEquals(7.0f, v, 1e-6f, "All elements should be 7.0");
        }
    }

    @Test
    void dtypeStaticAccessors() {
        // Method-style access
        DType fp32 = DType.fp32();
        assertNotNull(fp32);
        assertEquals("Float32", fp32.getName());

        DType fp16 = DType.fp16();
        assertNotNull(fp16);

        DType int32 = DType.int32();
        assertNotNull(int32);
    }

    @Test
    void dtypeStaticAccessorsReturnSameInstances() {
        // Repeated calls should return the same singleton instances
        assertSame(DType.fp32(), DType.fp32(),
                "fp32() should return the same instance on repeated calls");
        assertSame(DType.fp16(), DType.fp16(),
                "fp16() should return the same instance on repeated calls");
        assertSame(DType.int32(), DType.int32(),
                "int32() should return the same instance on repeated calls");
    }
}
