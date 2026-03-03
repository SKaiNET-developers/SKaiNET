# Build Help

## Dokka API Documentation

SKaiNET uses [Dokka 2.1.0](https://github.com/Kotlin/dokka) to generate API reference documentation across all public library modules. A shared convention plugin (`sk.ainet.dokka`) standardises the configuration.

### Generating docs locally

**Single module:**

```bash
./gradlew :skainet-lang:skainet-lang-core:dokkaGeneratePublicationHtml
```

Output: `skainet-lang/skainet-lang-core/build/dokka/html/`

**Aggregated (all modules):**

```bash
./gradlew dokkaGenerate
```

Output: `build/dokka/html/index.html`

### Convention plugin details

The `sk.ainet.dokka` precompiled script plugin (`build-logic/convention/src/main/kotlin/sk.ainet.dokka.gradle.kts`) applies `org.jetbrains.dokka` and configures:

- **moduleName** from `project.name`
- **moduleVersion** from the `VERSION_NAME` Gradle property
- **Documented visibilities:** public only
- **Suppressed generated files:** KSP-generated code is excluded
- **Suppressed native source sets:** `iosArm64Main`, `iosSimulatorArm64Main`, `macosArm64Main`, `linuxX64Main`, `linuxArm64Main` are suppressed because Dokka 2.x cannot translate native cinterop symbols
- **Source links** pointing to the GitHub repository

### Modules with Dokka enabled

The plugin is applied to 21 library modules:

| Group | Modules |
|-------|---------|
| skainet-lang | `skainet-lang-core`, `skainet-lang-models`, `skainet-lang-ksp-annotations`, `skainet-kan`, `skainet-lang-dag` |
| skainet-compile | `skainet-compile-core`, `skainet-compile-dag`, `skainet-compile-json`, `skainet-compile-hlo`, `skainet-compile-c` |
| skainet-backends | `skainet-backend-cpu` |
| skainet-data | `skainet-data-api`, `skainet-data-transform`, `skainet-data-simple`, `skainet-data-media` |
| skainet-io | `skainet-io-core`, `skainet-io-gguf`, `skainet-io-image`, `skainet-io-onnx`, `skainet-io-safetensors` |
| Other | `skainet-pipeline`, `skainet-model-yolo` |

**Excluded:** `skainet-bom` (no source), `skainet-starter-jvm`, `skainet-apps/*`, `skainet-test/*`, benchmarks, and `skainet-lang-ksp-processor` (internal).

### Root-level aggregation

The root `build.gradle.kts` applies the Dokka plugin directly (not `apply false`) and declares `dokka(project(...))` dependencies for all 21 modules. Running `./gradlew dokkaGenerate` at the root produces a unified API reference that includes every module under a single `SKaiNET` namespace. The root `README.md` is included as the landing page.

### KSP interaction

`skainet-lang-core` and `skainet-lang-dag` use KSP to generate source code. Their build files include:

```kotlin
tasks.matching { it.name.startsWith("dokka") }.configureEach {
    dependsOn("kspCommonMainKotlinMetadata")
}
```

This ensures KSP-generated sources are available before Dokka runs.

### GitHub Pages deployment

The workflow `.github/workflows/dokka-pages.yml` runs on push to `main` (and manually via `workflow_dispatch`). It:

1. Checks out the repo
2. Sets up JDK 25
3. Runs `./gradlew dokkaGenerate`
4. Uploads the `build/dokka/html` directory as a Pages artifact
5. Deploys to GitHub Pages using `actions/deploy-pages@v4`

**Prerequisite:** The repository must have Pages configured to deploy from GitHub Actions (Settings > Pages > Source: "GitHub Actions").

### Operator docs (unchanged)

The existing operator documentation pipeline (`./gradlew generateDocs`) is unrelated to Dokka and continues to work as before.
