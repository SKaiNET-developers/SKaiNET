package sk.ainet.apps.kgemma.cli

/**
 * Native CLI entry point for kgemma.
 *
 * Note: Full native CLI support is planned for a future release.
 * Currently, use the JVM version for full functionality:
 *   ./gradlew :skainet-apps:skainet-kgemma:jvmRun --args='<model> <prompt>'
 */
fun main(args: Array<String>) {
    println("kgemma - Kotlin Multiplatform Gemma 3n Runtime")
    println()
    println("Native CLI support is currently limited.")
    println("For full functionality, use the JVM version:")
    println()
    println("  ./gradlew :skainet-apps:skainet-kgemma:jvmRun \\")
    println("      --args='<model-path> \"<prompt>\" [steps] [temperature]'")
    println()
    println("Supported model formats:")
    println("  - GGUF: path/to/model.gguf")
    println("  - SafeTensors: path/to/model/ (directory with model.safetensors.index.json)")
    println()
    println("Example:")
    println("  ./gradlew :skainet-apps:skainet-kgemma:jvmRun \\")
    println("      --args='models/ \"Hello, how are you?\" 32 0.8'")
}
