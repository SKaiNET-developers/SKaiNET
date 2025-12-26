package sk.ainet.apps.knanogpt

class CharTokenizer(text: String) : Tokenizer<Int> {
    private val chars: List<Char> = text.toSet().sorted()
    private val stoi: Map<Char, Int> = chars.mapIndexed { index, c -> c to index }.toMap()
    private val itos: Map<Int, Char> = chars.mapIndexed { index, c -> index to c }.toMap()

    override fun encode(text: String): List<Int> {
        // Tokenizes the given text into tokens.
        return text.map { c -> stoi[c] ?: error("Character not found") }
    }

    override fun decode(tokens: List<Int>): String {
        // Reconstructs the text from the tokens.
        return tokens.map { i -> itos[i] ?: error("Token not found") }.joinToString("")
    }
}
