'use strict'

/*
 * Offline Mermaid block processor for Asciidoctor.js / Antora.
 *
 * Replaces asciidoctor-kroki's dependency on a Kroki HTTP server
 * (kroki.io or a local container — GET URL length limits, 400s on
 * large diagrams, no offline fallback) with a direct, synchronous
 * invocation of `mmdc` (@mermaid-js/mermaid-cli) baked into the
 * markup-antora image at /opt/antora. Nothing leaves the container.
 *
 * For every `[mermaid]\n----\n...\n----` (or literal `....`) block:
 *   1. hash the source (md5) and consult the cache
 *   2. on miss: write source to a temp file, run mmdc to produce SVG,
 *      read it back, store in the cache
 *   3. inline the SVG via a `pass` block so Asciidoctor emits raw SVG
 *      straight into the HTML output
 *
 * Caching (merged from the Daily-StandAPP pipeline extension):
 *   - In-memory Map, keyed by content hash, dedupes identical diagrams
 *     across every page in a single build run.
 *   - If MERMAID_CACHE_DIR is set, the SVG is also persisted there as
 *     <hash>.svg, so unchanged diagrams survive across build runs
 *     (point it at a host-mounted, writable dir to benefit).
 *
 * On render failure we degrade to a literal block containing the
 * original source plus the error — matching asciidoctor-kroki's mode.
 *
 * Register it from the Antora playbook:
 *   asciidoc:
 *     extensions:
 *       - /opt/antora/local-mermaid-extension.js
 */

const { execSync } = require('child_process')
const { mkdtempSync, writeFileSync, readFileSync, rmSync,
        existsSync, mkdirSync } = require('fs')
const { tmpdir } = require('os')
const { join } = require('path')
const { createHash } = require('crypto')

// Absolute paths baked into /opt/antora at image build time. These must
// match the Dockerfile that installs mermaid-cli and the puppeteer config.
const MMDC_BIN = '/opt/antora/node_modules/.bin/mmdc'
const PUPPETEER_CONFIG = '/opt/antora/puppeteer-config.json'

// Optional cross-run disk cache.
const DISK_CACHE_DIR = process.env.MERMAID_CACHE_DIR || null

// In-process cache: hash -> svg. Dedupes within a single build run.
const memCache = new Map()

function hashOf (source) {
  return createHash('md5').update(source).digest('hex').slice(0, 16)
}

function renderMermaidToSvg (source) {
  const key = hashOf(source)

  // 1. in-memory hit
  if (memCache.has(key)) return memCache.get(key)

  // 2. disk hit
  let diskPath = null
  if (DISK_CACHE_DIR) {
    diskPath = join(DISK_CACHE_DIR, `${key}.svg`)
    if (existsSync(diskPath)) {
      const cached = readFileSync(diskPath, 'utf8')
      memCache.set(key, cached)
      return cached
    }
  }

  // 3. miss — render with mermaid-cli
  const dir = mkdtempSync(join(tmpdir(), 'markup-antora-mm-'))
  const inputPath = join(dir, 'in.mmd')
  const outputPath = join(dir, 'out.svg')
  writeFileSync(inputPath, source, 'utf8')
  try {
    execSync(
      `${MMDC_BIN} -i ${inputPath} -o ${outputPath} -p ${PUPPETEER_CONFIG} --quiet`,
      { stdio: ['ignore', 'ignore', 'pipe'] }
    )
    const svg = readFileSync(outputPath, 'utf8')
    memCache.set(key, svg)
    if (diskPath) {
      try {
        if (!existsSync(DISK_CACHE_DIR)) mkdirSync(DISK_CACHE_DIR, { recursive: true })
        writeFileSync(diskPath, svg, 'utf8')
      } catch (_) { /* cache is best-effort */ }
    }
    return svg
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
