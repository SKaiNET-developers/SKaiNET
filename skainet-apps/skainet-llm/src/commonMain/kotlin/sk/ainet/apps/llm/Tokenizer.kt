package sk.ainet.apps.llm

interface Tokenizer {
    fun encode(text: String): IntArray
    fun decode(tokens: IntArray): String
    fun decode(token: Int): String
}
