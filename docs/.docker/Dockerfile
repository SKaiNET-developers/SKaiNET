FROM node:20-alpine

LABEL org.opencontainers.image.title="SKaiNET Antora" \
      org.opencontainers.image.description="Antora site generator with direct local Mermaid rendering (no Kroki round trip)" \
      org.opencontainers.image.source="https://github.com/SKaiNET-developers/SKaiNET"

# Chromium for mermaid-cli (puppeteer)
RUN apk add --no-cache chromium font-noto

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser \
    PUPPETEER_SKIP_DOWNLOAD=true

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

# Verify mermaid-cli works end to end at image build time.
RUN echo 'graph TD; A-->B;' > /tmp/test.mmd \
    && npx mmdc -i /tmp/test.mmd -o /tmp/test.svg -p /opt/antora/puppeteer-config.json \
    && rm /tmp/test.mmd /tmp/test.svg

ENTRYPOINT ["/opt/antora/node_modules/.bin/antora"]
CMD ["--stacktrace", "antora-playbook.yml"]
