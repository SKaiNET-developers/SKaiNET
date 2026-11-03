package sk.ainet.frontend.tokenizer

import java.util.regex.Pattern

// Android already uses Unicode classes and ignores this flag (available since its minSdk 24).
internal actual fun compileTokenizerSplitRegex(pattern: String): Regex =
    Pattern.compile(pattern, Pattern.UNICODE_CHARACTER_CLASS).toRegex()
