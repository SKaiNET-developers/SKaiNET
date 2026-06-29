# Docs Antora image (`docs/.docker`)

Self-contained Antora image used to build this repo's documentation site.
It is the consolidated "markup-antora" image — one definition shared across
the SKaiNET docs projects — vendored here until the public registry image
is published (after which this `Dockerfile` collapses to a single `FROM`).

## Features

- **Offline Mermaid** — every `[mermaid]` block is rendered to **inline SVG**
  at build time by `mermaid-cli` (Alpine Chromium + Puppeteer) via the baked-in
  `local-mermaid-extension.js` Asciidoctor block processor. No Kroki server,
  no `kroki.io`, no network — at build time *or* view time. Removes the
  asciidoctor-kroki 4 KB GET-URL limit that rejected large diagrams.
- **Diagram caching** — content-hash, in-memory + optional on-disk
  (`MERMAID_CACHE_DIR`); identical diagrams render once.
- **Rootless-safe** — runs under `--user $(id -u):$(id -g)` without the
  Chromium crashpad / cosmiconfig `EACCES` failures (`HOME=/tmp`, build-time
  cleanup of root-owned `/tmp` dirs).
- **Build-time smoke test** — a broken image fails `docker build`, not your
  first render.
- **Offline extras** — `@antora/lunr-extension` (search), a pre-baked Antora
  UI bundle, and MathJax es5 for LaTeX are available in the image.
- **Kroki escape hatch** — `asciidoctor-kroki` is installed (unused here) for
  other diagram types if ever needed.
- Full Alpine font set (`font-noto`, `font-noto-emoji`, `ttf-freefont`, …) so
  diagram labels, emoji and CJK render correctly.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | The consolidated image definition (build context = this dir). |
| `local-mermaid-extension.js` | Offline Mermaid block processor; baked to `/opt/antora/`. |
| `puppeteer-config.json` | Chromium flags for mermaid-cli; baked to `/opt/antora/`. |

The playbook wires the extension via
`asciidoc.extensions: [ /opt/antora/local-mermaid-extension.js ]`.

## Usage

Build the image (context is this directory):

```bash
docker build -t skainet-antora:local -f docs/.docker/Dockerfile docs/.docker/
```

Render the site (run from the repo root; mount the repo at `/antora`, run as
your user so output isn't root-owned):

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/antora" \
  --workdir /antora/docs \
  skainet-antora:local \
  --stacktrace antora-playbook.yml

# Output: docs/build/site/index.html
```

This is exactly what `.github/workflows/docs.yml` does in CI — it builds the
image from this directory and runs the container the same way.

Write diagrams as normal Asciidoctor blocks:

```adoc
[mermaid]
----
graph TD; A-->B; B-->C;
----
```
