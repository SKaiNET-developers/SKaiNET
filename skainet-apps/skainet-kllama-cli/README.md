# SKaiNET KLlama CLI

A simple command-line interface for running LLM models using the SKaiNET KLlama library.

For a comprehensive guide on how to use KLlama, including embedding it into your own applications, please refer to the [Getting Started Guide](../../docs/kllama-getting-started.md).

## 🚀 Quick Start

### Building

To build the executable fat JAR:

```bash
./gradlew :skainet-apps:skainet-kllama-cli:shadowJar
```

### Running

```bash
java --enable-preview --add-modules jdk.incubator.vector -jar skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar <model_path> [tokenizer_path] "<prompt>"
```

See [docs/kllama-getting-started.md](../../docs/kllama-getting-started.md) for more details.
