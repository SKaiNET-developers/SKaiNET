'use strict'

/*
 * Local mermaid block processor for Asciidoctor.js.
 *
 * Replaces the asciidoctor-kroki dependency on kroki.io (and its
 * GET URL length limit / 400 rejections on large diagrams) with a
 * direct invocation of `mmdc` — the @mermaid-js/mermaid-cli binary
 * that the SKaiNET Antora Docker image already bakes in for its
 * Chromium-backed Puppeteer rendering path.
 *
 * The extension is registered via the Antora playbook's
 * `asciidoc.extensions` list and gets passed the Asciidoctor.js
 * `registry` object. For every `[mermaid]\n----\n...\n----` block
 * in any page, we:
 *
 *   1. write the source to a temp file
 *   2. exec `mmdc -i in.mmd -o out.svg -p puppeteer-config.json`
 *      (synchronous — Antora processes one page at a time and the
 *      mermaid-cli call is fast enough that sync is fine)
 *   3. read the produced SVG
 *   4. inline it via a `pass` block so Asciidoctor emits the raw
 *      SVG markup straight into the HTML output
 *
 * On render failure we fall back to a literal block containing
 * the original source plus the error message, matching the
 * degradation mode asciidoctor-kroki uses.
 */

const { execSync } = require('child_process')
const { mkdtempSync, writeFileSync, readFileSync, rmSync } = require('fs')
const { tmpdir } = require('os')
const { join } = require('path')

// Absolute paths baked into /opt/antora at image build time.
// These have to match the Dockerfile that installs mermaid-cli and
// writes the puppeteer config.
const MMDC_BIN = '/opt/antora/node_modules/.bin/mmdc'
const PUPPETEER_CONFIG = '/opt/antora/puppeteer-config.json'

function renderMermaidToSvg (source) {
  const dir = mkdtempSync(join(tmpdir(), 'skainet-mm-'))
  const inputPath = join(dir, 'in.mmd')
  const outputPath = join(dir, 'out.svg')
  writeFileSync(inputPath, source, 'utf8')
  try {
    execSync(
      `${MMDC_BIN} -i ${inputPath} -o ${outputPath} -p ${PUPPETEER_CONFIG} --quiet`,
      { stdio: ['ignore', 'ignore', 'pipe'] }
    )
    return readFileSync(outputPath, 'utf8')
  } finally {
    try { rmSync(dir, { recursive: true, force: true }) } catch (_) { /* noop */ }
  }
}

function mermaidBlockFactory () {
  return function () {
    const self = this
    self.named('mermaid')
    self.onContext(['listing', 'literal'])
    self.process((parent, reader, attrs) => {
      const source = reader.$read()
      try {
        const svg = renderMermaidToSvg(source)
        return self.createBlock(parent, 'pass', svg, attrs)
      } catch (err) {
        const logger = parent.getDocument().getLogger()
        logger.warn(`local-mermaid-extension: failed to render block — ${err.message}`)
        const role = attrs.role
        attrs.role = role ? `${role} mermaid-error` : 'mermaid-error'
        return self.createBlock(
          parent,
          'literal',
          `Error rendering mermaid diagram:\n${err.message}\n\n${source}`,
          attrs
        )
      }
    })
  }
}

module.exports.register = function register (registry) {
  if (typeof registry.register === 'function') {
    registry.register(function () {
      this.block('mermaid', mermaidBlockFactory())
    })
  } else if (typeof registry.block === 'function') {
    registry.block('mermaid', mermaidBlockFactory())
  }
}
