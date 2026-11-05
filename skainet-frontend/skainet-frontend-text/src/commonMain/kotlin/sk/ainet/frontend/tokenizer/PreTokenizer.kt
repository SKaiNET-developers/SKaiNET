package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.frontend.UnsupportedFrontEndException

/**
 * The `tokenizer.json#pre_tokenizer` step: cuts a normalised segment into the chunks the model
 * (BPE merges, WordPiece lookup, Unigram lattice) runs over. Each implementation returns the
 * chunks in order; concatenated they rebuild the input exactly, except for [WhitespaceSplit] /
 * [Whitespace] / [Bert], which drop whitespace by design.
 */
public sealed interface PreTokenizer {
    public fun split(text: String): List<String>
    public val name: String

    /** No split: the whole segment is one chunk. */
    public object None : PreTokenizer {
        override fun split(text: String): List<String> = if (text.isEmpty()) emptyList() else listOf(text)
        override val name: String get() = "none"
    }

    public class Sequence(public val steps: List<PreTokenizer>) : PreTokenizer {
        override fun split(text: String): List<String> =
            steps.fold(listOf(text)) { chunks, p -> chunks.flatMap { p.split(it) } }
        override val name: String get() = steps.joinToString("+", prefix = "[", postfix = "]") { it.name }
    }

    /**
     * Regex split keeping the matches (`behavior: Isolated`, the GPT-2 / Qwen shape). Other
     * behaviours (`Removed`, `MergedWithPrevious`, …) are accepted for the common cases.
     */
    public class Split(public val pattern: String, public val behavior: String = "Isolated", public val invert: Boolean = false) : PreTokenizer {
        private val regex = compileTokenizerSplitRegex(pattern)
        override fun split(text: String): List<String> {
            if (text.isEmpty()) return emptyList()
            val out = ArrayList<String>()
            var last = 0
            for (m in regex.findAll(text)) {
                if (m.range.isEmpty()) continue
                val between = text.substring(last, m.range.first)
                when (behavior) {
                    "Removed" -> { if (between.isNotEmpty()) out.add(between) }
                    "MergedWithPrevious" -> { if (between.isNotEmpty()) out.add(between + m.value) else out.add(m.value) }
                    "MergedWithNext" -> { out.add(between + m.value) }
                    else -> { if (between.isNotEmpty()) out.add(between); out.add(m.value) }
                }
                last = m.range.last + 1
            }
            val tail = text.substring(last)
            if (tail.isNotEmpty()) {
                if (behavior == "MergedWithNext" && out.isNotEmpty()) out[out.lastIndex] = out.last() + tail else out.add(tail)
            }
            return if (invert) listOf(text) else out
        }
        override val name: String get() = "split"
    }

    /**
     * GPT-2 byte-level pre-tokenization: the GPT-2 regex when [useRegex], optional prefix space,
     * then every chunk mapped through [ByteToUnicode]. Chunks come back in byte-level alphabet
     * form, which is what the BPE merges are written in.
     */
    public class ByteLevel(public val addPrefixSpace: Boolean = false, public val useRegex: Boolean = true) : PreTokenizer {
        override fun split(text: String): List<String> {
            if (text.isEmpty()) return emptyList()
            val t = if (addPrefixSpace && !text.startsWith(' ')) " $text" else text
            val chunks = if (useRegex) GPT2_REGEX.findAll(t).map { it.value }.filter { it.isNotEmpty() }.toList() else listOf(t)
            return chunks.map { ByteToUnicode.encode(it) }
        }
        override val name: String get() = "byte-level"
    }

    /**
     * SentencePiece-style `Metaspace`: spaces become [replacement], a prefix is prepended per
     * [prependScheme] (`always`, `first`, `never`), and with [splitOnReplacement] the text is cut
     * so that every chunk starts at a word boundary.
     */
    public class Metaspace(
        public val replacement: Char = '▁',
        public val prependScheme: String = "always",
        public val splitOnReplacement: Boolean = true,
    ) : PreTokenizer {
        override fun split(text: String): List<String> {
            if (text.isEmpty()) return emptyList()
            var t = text.replace(' ', replacement)
            if (prependScheme != "never" && !t.startsWith(replacement)) t = replacement + t
            if (!splitOnReplacement) return listOf(t)
            val out = ArrayList<String>()
            var start = 0
            for (i in 1 until t.length) {
                if (t[i] == replacement) { out.add(t.substring(start, i)); start = i }
            }
            out.add(t.substring(start))
            return out
        }
        override val name: String get() = "metaspace"
    }

    /** Split on runs of whitespace, dropping the whitespace. */
    public object WhitespaceSplit : PreTokenizer {
        override fun split(text: String): List<String> = text.split(WS).filter { it.isNotEmpty() }
        override val name: String get() = "whitespace-split"
    }

    /** `\w+|[^\w\s]+`, dropping whitespace. */
    public object Whitespace : PreTokenizer {
        private val regex = Regex("\\w+|[^\\w\\s]+")
        override fun split(text: String): List<String> = regex.findAll(text).map { it.value }.toList()
        override val name: String get() = "whitespace"
    }

    /** The reference `BertPreTokenizer`: whitespace split, then every punctuation character on its own. */
    public object Bert : PreTokenizer {
        override fun split(text: String): List<String> {
            val out = ArrayList<String>()
            val sb = StringBuilder()
            for (ch in text) {
                when {
                    ch.isWhitespace() -> { if (sb.isNotEmpty()) { out.add(sb.toString()); sb.clear() } }
                    isPunctuation(ch) -> { if (sb.isNotEmpty()) { out.add(sb.toString()); sb.clear() }; out.add(ch.toString()) }
                    else -> sb.append(ch)
                }
            }
            if (sb.isNotEmpty()) out.add(sb.toString())
            return out
        }
        override val name: String get() = "bert"

        internal fun isPunctuation(ch: Char): Boolean {
            val code = ch.code
            if (code in 33..47 || code in 58..64 || code in 91..96 || code in 123..126) return true
            return when (ch.category) {
                CharCategory.OTHER_PUNCTUATION, CharCategory.DASH_PUNCTUATION, CharCategory.START_PUNCTUATION,
                CharCategory.END_PUNCTUATION, CharCategory.CONNECTOR_PUNCTUATION, CharCategory.INITIAL_QUOTE_PUNCTUATION,
                CharCategory.FINAL_QUOTE_PUNCTUATION -> true
                else -> false
            }
        }
    }

    /** Every punctuation character becomes its own chunk (`behavior: Isolated`). */
    public object Punctuation : PreTokenizer {
        override fun split(text: String): List<String> {
            val out = ArrayList<String>()
            val sb = StringBuilder()
            for (ch in text) {
                if (Bert.isPunctuation(ch)) { if (sb.isNotEmpty()) { out.add(sb.toString()); sb.clear() }; out.add(ch.toString()) } else sb.append(ch)
            }
            if (sb.isNotEmpty()) out.add(sb.toString())
            return out
        }
        override val name: String get() = "punctuation"
    }

    /** Digit runs (or single digits with [individualDigits]) become their own chunks. */
    public class Digits(public val individualDigits: Boolean = false) : PreTokenizer {
        override fun split(text: String): List<String> {
            val out = ArrayList<String>()
            val sb = StringBuilder()
            var sbDigits = false
            for (ch in text) {
                val d = ch.isDigit()
                if (sb.isNotEmpty() && (d != sbDigits || (d && individualDigits))) { out.add(sb.toString()); sb.clear() }
                sb.append(ch); sbDigits = d
            }
            if (sb.isNotEmpty()) out.add(sb.toString())
            return out
        }
        override val name: String get() = "digits"
    }

    public companion object {
        /** The GPT-2 pre-tokenization regex (Karpathy / Hugging Face). */
        public val GPT2_REGEX: Regex = Regex("'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

        private val WS = Regex("\\s+")

        /** Build from the `pre_tokenizer` element of a tokenizer.json root (`null` → [None]). */
        public fun fromJson(element: JsonElement?): PreTokenizer {
            if (element == null) return None
            val obj = element as? JsonObject ?: return None
            val type = obj["type"]?.jsonPrimitive?.content
                ?: throw UnsupportedFrontEndException("tokenizer.json pre_tokenizer without a type: $obj")
            return when (type) {
                "Sequence" -> Sequence(obj["pretokenizers"]?.jsonArray.orEmpty().map { fromJson(it) })
                "ByteLevel" -> ByteLevel(
                    addPrefixSpace = obj["add_prefix_space"]?.jsonPrimitive?.booleanOrNull ?: false,
                    useRegex = obj["use_regex"]?.jsonPrimitive?.booleanOrNull ?: true,
                )
                "Split" -> {
                    val pattern = obj["pattern"]?.jsonObject
                        ?: throw UnsupportedFrontEndException("tokenizer.json Split pre_tokenizer without a pattern")
                    val regex = pattern["Regex"]?.jsonPrimitive?.content
                        ?: pattern["String"]?.jsonPrimitive?.content?.let { Regex.escape(it) }
                        ?: throw UnsupportedFrontEndException("tokenizer.json Split pre_tokenizer: unknown pattern kind $pattern")
                    Split(regex, obj["behavior"]?.jsonPrimitive?.content ?: "Isolated", obj["invert"]?.jsonPrimitive?.booleanOrNull ?: false)
                }
                "Metaspace" -> {
                    val replacement = obj["replacement"]?.jsonPrimitive?.content?.firstOrNull() ?: '▁'
                    val scheme = obj["prepend_scheme"]?.jsonPrimitive?.content
                        ?: if (obj["add_prefix_space"]?.jsonPrimitive?.booleanOrNull == false) "never" else "always"
                    Metaspace(replacement, scheme, obj["split"]?.jsonPrimitive?.booleanOrNull ?: true)
                }
                "WhitespaceSplit" -> WhitespaceSplit
                "Whitespace" -> Whitespace
                "BertPreTokenizer" -> Bert
                "Punctuation" -> Punctuation
                "Digits" -> Digits(obj["individual_digits"]?.jsonPrimitive?.booleanOrNull ?: false)
                else -> throw UnsupportedFrontEndException("tokenizer.json pre_tokenizer type '$type' is not supported")
            }
        }

        /** Does this pre-tokenizer (or any member of a sequence) produce byte-level alphabet chunks? */
        public fun isByteLevel(p: PreTokenizer): Boolean = when (p) {
            is ByteLevel -> true
            is Sequence -> p.steps.any { isByteLevel(it) }
            else -> false
        }

        /** The Metaspace member of this pre-tokenizer, if any. */
        public fun metaspace(p: PreTokenizer): Metaspace? = when (p) {
            is Metaspace -> p
            is Sequence -> p.steps.firstNotNullOfOrNull { metaspace(it) }
            else -> null
        }
    }
}
