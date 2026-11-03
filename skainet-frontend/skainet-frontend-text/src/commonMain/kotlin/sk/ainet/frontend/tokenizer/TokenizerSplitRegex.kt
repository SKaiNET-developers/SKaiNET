package sk.ainet.frontend.tokenizer

/**
 * Compile a checkpoint's Split expression without rewriting its pattern text.
 * JVM enables Unicode shorthand classes, matching the tokenizer.json reference behavior.
 * Android already uses Unicode classes; other platforms retain their Kotlin Regex implementation.
 */
internal expect fun compileTokenizerSplitRegex(pattern: String): Regex
