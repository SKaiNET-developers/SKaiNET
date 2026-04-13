package sk.ainet.java;

import org.junit.jupiter.api.Test;

import sk.ainet.compile.hlo.StableHloConverter;
import sk.ainet.compile.hlo.StableHloConverterFactory;
import sk.ainet.io.tokenizer.TokenizerFactory;
import sk.ainet.io.tokenizer.UnsupportedTokenizerException;
import sk.ainet.lang.tensor.ops.TensorSpec;
import sk.ainet.lang.tensor.ops.TensorSpecs;
import sk.ainet.lang.tensor.storage.TensorEncoding;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Java consumer smoke tests for the Kotlin surfaces polished in the
 * 0.19.0 release for Java-first-citizenship (#400). Each test is
 * deliberately close to the call sites a Java consumer of the BOM
 * would write so the patterns are self-documenting.
 *
 * If any of these lose their clean static form on a future Kotlin
 * refactor (e.g. `@JvmStatic` dropped), the tests fail at compile
 * time rather than at bytecode-verification time in production.
 */
class ReleaseApiJavaTest {

    // --- StableHloConverterFactory -----------------------------------------

    /**
     * The converter factory must be reachable via the idiomatic
     * static form, not through the Kotlin object's INSTANCE marker.
     * Written as a compile-time smoke test — if someone drops the
     * @JvmStatic annotations this fails to compile before any
     * assertion runs.
     */
    @Test
    void stableHloConverterFactoryIsStatic() {
        StableHloConverter basic = StableHloConverterFactory.createBasic();
        StableHloConverter extended = StableHloConverterFactory.createExtended();
        StableHloConverter fast = StableHloConverterFactory.createFast();

        assertNotNull(basic, "createBasic() must return a non-null converter");
        assertNotNull(extended, "createExtended() must return a non-null converter");
        assertNotNull(fast, "createFast() must return a non-null converter");
    }

    // --- TokenizerFactory --------------------------------------------------

    /**
     * TokenizerFactory's static form is the entry point Java consumers
     * hit when they load a GGUF or HuggingFace tokenizer.json. The
     * call shape must stay clean across releases.
     *
     * We pass an empty GGUF field map and expect an
     * UnsupportedTokenizerException — the point is to prove the
     * factory is dispatched via static, not to actually tokenize.
     */
    @Test
    void tokenizerFactoryFromGgufIsStatic() {
        Map<String, Object> emptyFields = Collections.emptyMap();
        assertThrows(
            UnsupportedTokenizerException.class,
            () -> TokenizerFactory.fromGguf(emptyFields),
            "empty GGUF metadata map must trip UnsupportedTokenizerException"
        );
    }

    @Test
    void tokenizerFactoryFromTokenizerJsonIsStatic() {
        String emptyJson = "{}";
        assertThrows(
            UnsupportedTokenizerException.class,
            () -> TokenizerFactory.fromTokenizerJson(emptyJson),
            "tokenizer.json with no model.type must trip UnsupportedTokenizerException"
        );
    }

    // --- TensorSpecs (JvmName of TensorSpecEncoding.kt) --------------------

    /**
     * The TensorEncoding accessor helpers live on
     * skainet-lang-core/.../ops/TensorSpecEncoding.kt, which now
     * compiles to a class named TensorSpecs (via @file:JvmName).
     * Java callers access read / copy via static-method syntax.
     */
    @Test
    void tensorSpecsEncodingHelpers() {
        TensorSpec bare = new TensorSpec(
            /* name= */ "w",
            /* shape= */ List.of(8, 4),
            /* dtype= */ "FP32",
            /* requiresGrad= */ false,
            /* metadata= */ Collections.emptyMap()
        );

        // Reader: an un-annotated spec has a null encoding.
        assertNull(TensorSpecs.getTensorEncoding(bare),
            "a fresh TensorSpec must have no tensorEncoding");

        // Setter: returns a copy with the encoding attached.
        TensorSpec annotated = TensorSpecs.withTensorEncoding(
            bare, TensorEncoding.Q8_0.INSTANCE);
        assertNotNull(TensorSpecs.getTensorEncoding(annotated),
            "annotated spec must have a non-null tensorEncoding");
        assertSame(TensorEncoding.Q8_0.INSTANCE,
            TensorSpecs.getTensorEncoding(annotated),
            "annotated spec must carry the encoding we set");

        // The original is unchanged — data-class copy semantics.
        assertNull(TensorSpecs.getTensorEncoding(bare),
            "withTensorEncoding must not mutate the source spec");
    }
}
