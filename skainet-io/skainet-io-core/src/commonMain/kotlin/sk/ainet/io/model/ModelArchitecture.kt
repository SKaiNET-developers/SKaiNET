package sk.ainet.io.model

/**
 * Known model architectures that SKaiNET can load and run.
 */
public enum class ModelArchitecture {
    LLAMA,
    GEMMA,
    BERT,
    QWEN,
    YOLO,
    UNKNOWN;

    public companion object
}
