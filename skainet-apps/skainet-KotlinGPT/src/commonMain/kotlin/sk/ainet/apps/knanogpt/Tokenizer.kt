package sk.ainet.apps.knanogpt

interface Tokenizer<T> {

    fun encode(text: String): List<T>

    fun decode(tokens: List<T>): String
}