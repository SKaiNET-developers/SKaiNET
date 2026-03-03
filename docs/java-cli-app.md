# Building a Java CLI App with KLlama

This guide walks you through creating a standalone Java 21+ command-line application that loads a LLaMA model and generates text using the KLlama library.

## Prerequisites

- **JDK 21 or later** (required for Vector API and virtual threads)
- **Maven 3.8+** or **Gradle 8.4+**
- A GGUF model file (e.g., [TinyLlama-1.1B-Chat GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF))

---

## Project Setup

### Maven

Create a `pom.xml`:

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
                             http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>kllama-cli</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>

    <properties>
        <maven.compiler.source>21</maven.compiler.source>
        <maven.compiler.target>21</maven.compiler.target>
        <skainet.version>0.13.0</skainet.version>
    </properties>

    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>sk.ainet</groupId>
                <artifactId>skainet-bom</artifactId>
                <version>${skainet.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <dependencies>
        <!-- LLaMA inference -->
        <dependency>
            <groupId>sk.ainet</groupId>
            <artifactId>skainet-kllama-jvm</artifactId>
        </dependency>

        <!-- CPU backend (SIMD-accelerated) -->
        <dependency>
            <groupId>sk.ainet</groupId>
            <artifactId>skainet-backend-cpu-jvm</artifactId>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>21</source>
                    <target>21</target>
                    <compilerArgs>
                        <arg>--enable-preview</arg>
                    </compilerArgs>
                </configuration>
            </plugin>

            <!-- Run with: mvn compile exec:java -Dexec.args="..." -->
            <plugin>
                <groupId>org.codehaus.mojo</groupId>
                <artifactId>exec-maven-plugin</artifactId>
                <version>3.1.0</version>
                <configuration>
                    <mainClass>com.example.KLlamaCli</mainClass>
                    <jvmArgs>
                        <jvmArg>--enable-preview</jvmArg>
                        <jvmArg>--add-modules</jvmArg>
                        <jvmArg>jdk.incubator.vector</jvmArg>
                    </jvmArgs>
                </configuration>
            </plugin>

            <!-- Fat JAR for distribution -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-shade-plugin</artifactId>
                <version>3.5.1</version>
                <executions>
                    <execution>
                        <phase>package</phase>
                        <goals><goal>shade</goal></goals>
                        <configuration>
                            <transformers>
                                <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                                    <mainClass>com.example.KLlamaCli</mainClass>
                                </transformer>
                            </transformers>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
```

### Gradle

Create a `build.gradle` (Groovy DSL):

```groovy
plugins {
    id 'java'
    id 'application'
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories {
    mavenCentral()
}

dependencies {
    implementation platform('sk.ainet:skainet-bom:0.13.0')
    implementation 'sk.ainet:skainet-kllama-jvm'
    implementation 'sk.ainet:skainet-backend-cpu-jvm'
}

application {
    mainClass = 'com.example.KLlamaCli'
    applicationDefaultJvmArgs = [
        '--enable-preview',
        '--add-modules', 'jdk.incubator.vector'
    ]
}

tasks.withType(JavaCompile).configureEach {
    options.compilerArgs.add('--enable-preview')
}
```

---

## Source Code

Create `src/main/java/com/example/KLlamaCli.java`:

```java
package com.example;

import sk.ainet.apps.kllama.java.GenerationConfig;
import sk.ainet.apps.kllama.java.KLlamaJava;
import sk.ainet.apps.kllama.java.KLlamaSession;
import java.nio.file.Path;

public class KLlamaCli {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: kllama-cli <model.gguf> \"<prompt>\" [maxTokens] [temperature]");
            System.exit(1);
        }

        Path modelPath = Path.of(args[0]);
        String prompt = args[1];
        int maxTokens = args.length > 2 ? Integer.parseInt(args[2]) : 128;
        float temperature = args.length > 3 ? Float.parseFloat(args[3]) : 0.8f;

        GenerationConfig config = GenerationConfig.builder()
                .maxTokens(maxTokens)
                .temperature(temperature)
                .build();

        System.out.println("Loading model from " + modelPath + " ...");

        try (KLlamaSession session = KLlamaJava.loadGGUF(modelPath)) {
            // Stream tokens to stdout as they are generated
            session.generate(prompt, config, token -> System.out.print(token));
            System.out.println();
        }
    }
}
```

---

## Building and Running

### With Maven

```bash
# Run directly
mvn compile exec:java -Dexec.args="model.gguf 'Once upon a time' 128 0.7"

# Build fat JAR
mvn package

# Run from JAR
java --enable-preview --add-modules jdk.incubator.vector \
     -jar target/kllama-cli-1.0-SNAPSHOT.jar \
     model.gguf "Once upon a time" 128 0.7
```

### With Gradle

```bash
# Run directly
./gradlew run --args="model.gguf 'Once upon a time' 128 0.7"

# Build distribution
./gradlew installDist

# Run from distribution
./build/install/kllama-cli/bin/kllama-cli \
     model.gguf "Once upon a time" 128 0.7
```

---

## Loading SafeTensors Models

To load a HuggingFace model directory instead of GGUF, use `loadSafeTensors` and point to the directory containing `model.safetensors`, `config.json`, and `tokenizer.json`:

```java
try (KLlamaSession session = KLlamaJava.loadSafeTensors(Path.of("./my-llama-model/"))) {
    session.generate("Hello", config, token -> System.out.print(token));
    System.out.println();
}
```

---

## Async Generation

Use `generateAsync` to run generation on a virtual thread and get a `CompletableFuture`:

```java
import java.util.concurrent.CompletableFuture;

try (KLlamaSession session = KLlamaJava.loadGGUF(modelPath)) {
    CompletableFuture<String> future = session.generateAsync(
            "Explain quantum computing in one sentence",
            GenerationConfig.builder().maxTokens(64).build()
    );

    // Do other work while generation runs...

    String result = future.join();
    System.out.println(result);
}
```

You can also compose futures:

```java
session.generateAsync("Translate to French: Hello world")
       .thenAccept(translation -> System.out.println("Translation: " + translation))
       .exceptionally(ex -> { ex.printStackTrace(); return null; });
```

---

## Next Steps

- [Java LLM Inference Guide](java-llm-inference.md) — BERT embeddings, agent/tool-calling, and more.
- [Java Getting Started](java-getting-started.md) — tensor operations, full Maven/Gradle setup.
- [KLlama Library](../skainet-apps/skainet-kllama/README.md) — custom backends and Kotlin embedding.
