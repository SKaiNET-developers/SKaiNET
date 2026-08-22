# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities **privately** — do not open a public issue,
pull request, or discussion for a suspected vulnerability.

Use GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the issue, the affected component/version, and a reproduction if possible.

This opens a private advisory visible only to the maintainers. We will acknowledge
the report, investigate, and coordinate a fix and disclosure with you. Please give us
reasonable time to address the issue before any public disclosure.

If you are unable to use private reporting, contact a maintainer listed in the
repository metadata and request a private channel before sharing details.

## Supported versions

SKaiNET is pre-1.0 and evolving quickly. Security fixes are applied to the
**latest release** and the **`develop`** branch. Older versions are not maintained;
please upgrade to the latest version before reporting.

## Scope

In scope:

- The SKaiNET libraries published from this repository.
- Memory-safety, parsing, and deserialization issues in the model I/O readers
  (GGUF, SafeTensors, ONNX) when handling untrusted model files.
- Issues in generated export artifacts (e.g. Minerva/StableHLO) that could lead to
  unsafe code on a consumer's device.

Out of scope:

- Vulnerabilities in third-party dependencies — report those upstream (we will still
  bump the dependency once a fix is available).
- Denial of service from intentionally malformed inputs where the documented
  contract is "trusted input only."

## Handling dependency CVEs

Dependabot flags high/critical-severity advisories against this repo's Maven/JVM and
npm dependency graphs, including transitive ones no build script declares directly.

- **A transitive npm/Yarn dependency** (Kotlin/JS or Kotlin/Wasm): pinned via
  `sk.ainet.npm-pins` — see [Pinning npm Packages](docs/modules/ROOT/pages/contributing/build-from-source.adoc#pinning-npm-packages).
- **A transitive Maven/JVM dependency of the app's own graph**: pinned via
  `sk.ainet.maven-pins` — see [Pinning Maven/JVM Dependencies](docs/modules/ROOT/pages/contributing/build-from-source.adoc#pinning-mavenjvm-dependencies).
- **A transitive dependency of a *Gradle plugin's own classpath*** (AGP, Dokka, KSP,
  etc.) — neither mechanism above can reach these; see the "What this cannot fix"
  note in the Maven/JVM pinning doc linked above. These packages are build-time-only
  and are not present in anything SKaiNET publishes, so unless the vulnerable code
  path is actually reachable during a build, the alert is usually dismissed as
  tolerable risk with that reasoning recorded on the alert, rather than forcing a
  plugin version bump purely to silence the scanner. See
  [issue #1046](https://github.com/SKaiNET-developers/SKaiNET/issues/1046) for a
  worked example of this triage.

## Hardening and best practices

Broader open-source security posture (REUSE/OpenSSF Best Practices, SBOM, dependency
scanning) is tracked in the project's open-source best-practices work. See the
[Best Practices](https://www.bestpractices.dev/) program for the criteria we are
working toward.
