# markup-antora — one Antora image to use everywhere.
#
# Consolidates the best of the ~9 Antora images that grew across the
# projects tree. Design priorities, in order:
#   1. Mermaid rendered FULLY OFFLINE  — no Kroki server, no kroki.io,
#      no CDN at build time and none at view time. Diagrams are baked
#      to inline SVG by mermaid-cli (Alpine Chromium + Puppeteer)
#      through the local-mermaid-extension.js Asciidoctor block
#      processor, with content-hash caching so repeated diagrams render
#      once per build.
#   2. Works under `--user $(id -u):$(id -g)` (rootless) without the
#      Chromium crashpad / cosmiconfig EACCES failures.
#   3. Offline extras available but not forced: lunr full-text search,
#      a pre-baked Antora UI bundle, and MathJax es5 for LaTeX.
#   4. asciidoctor-kroki installed-but-unused as an escape hatch.
FROM node:20-alpine

LABEL org.opencontainers.image.title="markup-antora" \
      org.opencontainers.image.description="Universal Antora site generator with offline Mermaid (mermaid-cli), offline search (lunr), pre-baked UI bundle + MathJax. No Kroki, no CDN." \
      org.opencontainers.image.source="https://github.com/SKaiNET-developers/SKaiNET"

# Chromium for mermaid-cli (Puppeteer). Full font set so diagram labels,
# emoji and CJK render correctly (merged from the Daily-StandAPP image).
RUN apk add --no-cache \
      chromium \
      nss \
      freetype \
      harfbuzz \
      ttf-freefont \
      font-noto \
      font-noto-emoji \
      ca-certificates \
      git

# HOME=/tmp: Chromium's crashpad handler writes its database under $HOME
# and aborts with `--database is required` when the container runs as a
# non-root --user and $HOME falls back to `/` (no passwd entry, not
# writable). Same motivation as runtime.cache_dir in the playbook.
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser \
    PUPPETEER_SKIP_DOWNLOAD=true \
    HOME=/tmp

# Install Antora + tooling into /opt/antora (NOT /antora, which is where
# the project gets volume-mounted at run time).
#   - @mermaid-js/mermaid-cli  : offline diagram rendering (the point)
#   - @antora/lunr-extension   : offline full-text search
#   - asciidoctor-kroki        : escape hatch only; the playbook should
#                                use the local mermaid extension instead.
WORKDIR /opt/antora
RUN npm init -y && npm i --save-exact \
      @antora/cli@3.1 \
      @antora/site-generator@3.1 \
      @antora/lunr-extension@1.0.0-alpha.8 \
      @mermaid-js/mermaid-cli@11 \
      asciidoctor-kroki@0.18 \
    && npm cache clean --force

# Make installed modules resolvable even when the workdir is the mounted
# project (which has no node_modules of its own).
ENV NODE_PATH=/opt/antora/node_modules

# Mermaid-cli / Puppeteer config and the offline block processor, baked
# in at absolute paths the playbook can reference without mount gymnastics.
COPY puppeteer-config.json /opt/antora/puppeteer-config.json
COPY local-mermaid-extension.js /opt/antora/local-mermaid-extension.js

# --- Offline assets (available, not forced) -------------------------------

# Pre-download the default Antora UI bundle so sites build without hitting
# gitlab.com. Reference it from a playbook with:
#   ui:
#     bundle:
#       url: /opt/antora-ui/ui-bundle.zip
#       snapshot: true
RUN mkdir -p /opt/antora-ui \
    && wget -q -O /opt/antora-ui/ui-bundle.zip \
       "https://gitlab.com/antora/antora-ui-default/-/jobs/artifacts/HEAD/raw/build/ui-bundle.zip?job=bundle-stable"

# Pre-download MathJax es5 for offline LaTeX. Copy /opt/mathjax/es5 into a
# supplemental UI or reference it from your UI template for client-side math.
RUN mkdir -p /opt/mathjax \
    && npm pack mathjax@3 --pack-destination /tmp \
    && tar -xzf /tmp/mathjax-*.tgz -C /tmp \
    && cp -r /tmp/package/es5 /opt/mathjax/es5 \
    && rm -rf /tmp/mathjax-* /tmp/package

# --- Build-time smoke test + rootless cleanup -----------------------------

# Verify mermaid-cli works end to end so a broken image fails the build,
# not the user's first run. The cleanup also removes the mode-0700
# root-owned dirs (/tmp/.config/puppeteer, /tmp/.local/share/chromium,
# /tmp/.cache, /tmp/.npm) that Puppeteer/Chromium drop into $HOME during
# this run — leaving them would make cosmiconfig EACCES when the container
# is later launched with a non-root --user.
RUN echo 'graph TD; A-->B;' > /tmp/test.mmd \
    && /opt/antora/node_modules/.bin/mmdc \
         -i /tmp/test.mmd -o /tmp/test.svg \
         -p /opt/antora/puppeteer-config.json --quiet \
    && rm -rf /tmp/test.mmd /tmp/test.svg /tmp/.config /tmp/.local /tmp/.npm /tmp/.cache

WORKDIR /antora
ENTRYPOINT ["/opt/antora/node_modules/.bin/antora"]
CMD ["--stacktrace", "antora-playbook.yml"]
