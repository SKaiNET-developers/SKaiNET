package sk.ainet.lang.tensor

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class TensorIdTest {

    @Test
    fun canonicalAndLegacyForms() {
        val id = TensorId(listOf("model", "layers", "blk.3", "attn"), "q_proj.weight")
        assertEquals("model.layers.blk.3.attn.q_proj.weight", id.canonical)
        assertEquals(id.canonical, id.toString())
        assertEquals("model/layers/blk.3/attn", id.legacyPath())
        assertEquals("model|layers|blk.3|attn", id.legacyPath("|"))
        assertNull(id.discriminator)
    }

    @Test
    fun discriminatorAndViews() {
        val scores = TensorId(listOf("model", "layers[3]", "attn"), "scores", "step=17")
        assertEquals("model.layers[3].attn.scores#step=17", scores.canonical)
        assertEquals("model.layers[3].attn.scores", scores.withDiscriminator(null).canonical)
        assertEquals("kv.layers[3].k[1024..2048]", TensorId(listOf("kv", "layers[3]"), "k").view("1024..2048").canonical)
    }

    @Test
    fun parseInvertsCanonical() {
        val id = TensorId.parse("model.layers[3].attn.q_proj.weight#step=17")
        assertEquals(listOf("model", "layers[3]", "attn", "q_proj"), id.modulePath)
        assertEquals("weight", id.parameter)
        assertEquals("step=17", id.discriminator)
        assertEquals("model.layers[3].attn.q_proj.weight#step=17", id.canonical)
        // round trip as a string for every well-formed id
        for (s in listOf("weight", "a.b", "MLP.blk.0.attn.weight", "x.y#d")) assertEquals(s, TensorId.parse(s).canonical)
        assertEquals(TensorId(emptyList(), "weight"), TensorId.parse("weight"))
    }

    @Test
    fun equalityIsByCanonicalString() {
        val built = TensorId(listOf("MLP", "blk.0", "attn"), "weight")
        val parsed = TensorId.parse("MLP.blk.0.attn.weight")
        assertEquals(built, parsed)
        assertEquals(built.hashCode(), parsed.hashCode())
        assertEquals(listOf("MLP", "blk", "0", "attn"), parsed.modulePath) // structure differs, identity does not
        assertEquals("MLP/blk.0/attn", built.legacyPath())
        assertEquals(setOf(built), setOf(built, parsed))
    }

    @Test
    fun fromLegacyPath() {
        assertEquals(TensorId(listOf("MLP", "blk.0", "attn"), "weight"), TensorId.fromLegacyPath("MLP/blk.0/attn", "weight"))
        assertEquals(TensorId(emptyList(), "bias"), TensorId.fromLegacyPath(null, "bias"))
        assertEquals(TensorId(emptyList(), "bias"), TensorId.fromLegacyPath("", "bias"))
        assertEquals("MLP/blk.0/attn", TensorId.fromLegacyPath("MLP/blk.0/attn", "weight").legacyPath())
    }

    @Test
    fun validation() {
        assertFailsWith<IllegalArgumentException> { TensorId(listOf("a"), "") }
        assertFailsWith<IllegalArgumentException> { TensorId(listOf("a", ""), "w") }
        assertFailsWith<IllegalArgumentException> { TensorId.parse("") }
        assertFailsWith<IllegalArgumentException> { TensorId.parse("a..b") }
        assertFailsWith<IllegalArgumentException> { TensorId.parse(".w") }
    }
}
