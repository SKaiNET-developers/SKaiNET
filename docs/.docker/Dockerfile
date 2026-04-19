FROM node:20-alpine

LABEL org.opencontainers.image.title="SKaiNET Antora" \
      org.opencontainers.image.description="Antora site generator with direct local Mermaid rendering (no Kroki round trip)" \
      org.opencontainers.image.source="https://github.com/SKaiNET-developers/SKaiNET"

# Chromium for mermaid-cli (puppeteer)
RUN apk add --no-cache chromium font-noto

# HOME=/tmp: chromium's crashpad handler writes its database under $HOME and
# aborts with `chrome_crashpad_handler: --database is required` when the
# container runs as `--user $(id -u):$(id -g)` and $HOME falls back to `/`
# (no passwd entry, not writable). Same motivation as runtime.cache_dir in
# antora-playbook.yml.
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser \
    PUPPETEER_SKIP_DOWNLOAD=true \
    HOME=/tmp

# Install Antora + mermaid-cli into /opt/antora (not /antora which gets
# volume-mounted at run time). asciidoctor-kroki is intentionally NOT
# installed — it depends on a Kroki HTTP server (kroki.io or local)
# which returns 400 for large diagrams when using GET and has no
# offline fallback. We render mermaid directly via mermaid-cli through
# the local-mermaid-extension.js asciidoctor block processor.
WORKDIR /opt/antora
RUN npm init -y && npm i --save-exact \
      @antora/cli@3.1 \
      @antora/site-generator@3.1 \
      @mermaid-js/mermaid-cli@11 \
    && npm cache clean --force

# Make installed modules visible when workdir is the mounted project
ENV NODE_PATH=/opt/antora/node_modules

# Mermaid-cli config — used by the local-mermaid-extension to drive
# Puppeteer against the pre-installed Alpine Chromium.
RUN echo '{ \
  "executablePath": "/usr/bin/chromium-browser", \
  "args": ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] \
}' > /opt/antora/puppeteer-config.json

# Bake the local mermaid extension in at an absolute path so the
# Antora playbook can reference it without any volume-mount gymnastics.
COPY local-mermaid-extension.js /opt/antora/local-mermaid-extension.js

# Verify mermaid-cli works end to end at image build time. The cleanup
# also removes mode-0700 root-owned dirs (e.g. /tmp/.config/puppeteer,
# /tmp/.local/share/chromium) that puppeteer/chromium drop into $HOME
# during this run — leaving them in place would make cosmiconfig EACCES
# when the container is later launched with a non-root --user.
RUN echo 'graph TD; A-->B;' > /tmp/test.mmd \
    && npx mmdc -i /tmp/test.mmd -o /tmp/test.svg -p /opt/antora/puppeteer-config.json \
    && rm -rf /tmp/test.mmd /tmp/test.svg /tmp/.config /tmp/.local /tmp/.npm /tmp/.cache

ENTRYPOINT ["/opt/antora/node_modules/.bin/antora"]
CMD ["--stacktrace", "antora-playbook.yml"]
