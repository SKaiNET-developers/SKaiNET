package sk.ainet.java;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import sk.ainet.context.ExecutionContext;
import sk.ainet.lang.nn.Module;
import sk.ainet.lang.tensor.Tensor;
import sk.ainet.lang.types.DType;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the Java-friendly {@link SequentialModelBuilder} API.
 */
class ModelBuilderTest {

    private static ExecutionContext ctx;

    @BeforeAll
    static void setUp() {
        ctx = SKaiNET.context();
    }

    @Test
    void buildSimpleModel() {
        Module<?, ?> model = new SequentialModelBuilder(ctx)
                .input(4)
                .dense(8)
                .relu()
                .dense(2)
                .build();

        assertNotNull(model, "build() must return a non-null Module");
    }

    @Test
    void buildModelWithMultipleLayers() {
        Module<?, ?> model = new SequentialModelBuilder(ctx)
                .input(10)
                .dense(32)
                .relu()
                .dense(16)
                .relu()
                .dense(3)
                .build();

        assertNotNull(model);
    }

    @Test
    void buildModelWithSigmoid() {
        Module<?, ?> model = new SequentialModelBuilder(ctx)
                .input(4)
                .dense(8)
                .sigmoid()
                .dense(1)
                .build();

        assertNotNull(model);
    }

    @Test
    void buildModelWithSoftmax() {
        Module<?, ?> model = new SequentialModelBuilder(ctx)
                .input(4)
                .dense(8)
                .relu()
                .dense(3)
                .softmax()
                .build();

        assertNotNull(model);
    }

    @SuppressWarnings("unchecked")
    @Test
    void forwardPass() {
        Module<DType, Object> model = (Module<DType, Object>) (Module<?, ?>) new SequentialModelBuilder(ctx)
                .input(4)
                .dense(8)
                .relu()
                .dense(2)
                .build();

        // Create a [1, 4] input tensor (batch size 1, 4 features)
        Tensor<?, ?> input = SKaiNET.tensor(ctx, new int[]{1, 4}, DType.fp32(),
                new float[]{1.0f, 2.0f, 3.0f, 4.0f});

        @SuppressWarnings("unchecked")
        Tensor<DType, Object> typedInput = (Tensor<DType, Object>) input;
        Tensor<?, ?> output = model.forward(typedInput, ctx);

        assertNotNull(output, "Forward pass must produce a non-null output");
        // Output should have shape [1, 2] (batch=1, output features=2)
        assertArrayEquals(new int[]{1, 2}, output.getShape().getDimensions());
    }

    @Test
    void builderRequiresInputFirst() {
        SequentialModelBuilder builder = new SequentialModelBuilder(ctx);
        assertThrows(IllegalArgumentException.class, () -> builder.dense(8),
                "dense() without input() should throw IllegalArgumentException");
    }

    @Test
    void builderRequiresAtLeastOneLayer() {
        SequentialModelBuilder builder = new SequentialModelBuilder(ctx);
        assertThrows(IllegalArgumentException.class, builder::build,
                "build() with no layers should throw IllegalArgumentException");
    }

    @Test
    void explicitDtype() {
        Module<?, ?> model = new SequentialModelBuilder(ctx, DType.fp32())
                .input(4)
                .dense(2)
                .build();

        assertNotNull(model);
    }
}
