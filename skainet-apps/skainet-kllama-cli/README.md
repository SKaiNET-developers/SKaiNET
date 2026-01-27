# SKaiNET KLlama CLI

A simple command-line interface for running LLM models using the SKaiNET KLlama library.

## Building

To build the executable fat JAR, run the following command from the project root with Java 21 as the default JDK:

```bash
./gradlew :skainet-apps:skainet-kllama-cli:shadowJar
```

The resulting JAR will be located at:
`skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar`

## Running

To run the CLI, you need to use Java 21 or later and enable the Vector API incubator features.

### Usage

```bash
java --enable-preview --add-modules jdk.incubator.vector -jar skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar <model> [tokenizer] <prompt> [steps] [temperature]
```

### Parameters

- `<model>`: Path to the `.gguf` or `.bin` model file.
- `[tokenizer]`: Path to the `tokenizer.bin` (required for `.bin` models, optional for `.gguf` if embedded).
- `<prompt>`: The text prompt to generate from.
- `[steps]`: Number of tokens to generate (default: 64).
- `[temperature]`: Sampling temperature (default: 0.8).

### Example

Running with the provided TinyLlama model:

```bash
java --enable-preview --add-modules jdk.incubator.vector -jar skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar tinyllama-1.1b-q4.gguf "Once upon a time"
```
