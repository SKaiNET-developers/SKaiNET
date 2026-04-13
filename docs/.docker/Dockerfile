FROM node:20-alpine

LABEL org.opencontainers.image.title="SKaiNET Antora" \
      org.opencontainers.image.description="Antora site generator with built-in Mermaid rendering" \
      org.opencontainers.image.source="https://github.com/SKaiNET-developers/SKaiNET-transformers"

# Chromium for mermaid-cli (puppeteer)
RUN apk add --no-cache chromium font-noto

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser \
    PUPPETEER_SKIP_DOWNLOAD=true

# Install Antora + extensions to /opt/antora (not /antora which gets volume-mounted)
WORKDIR /opt/antora
RUN npm init -y && npm i --save-exact \
      @antora/cli@3.1 \
      @antora/site-generator@3.1 \
      asciidoctor-kroki@0.18 \
      @mermaid-js/mermaid-cli@11 \
    && npm cache clean --force

# Make installed modules visible when workdir is the mounted project
ENV NODE_PATH=/opt/antora/node_modules

# Mermaid-cli config
RUN echo '{ \
  "executablePath": "/usr/bin/chromium-browser", \
  "args": ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] \
}' > /opt/antora/puppeteer-config.json

# Verify mermaid works
RUN echo 'graph TD; A-->B;' > /tmp/test.mmd \
    && npx mmdc -i /tmp/test.mmd -o /tmp/test.svg -p /opt/antora/puppeteer-config.json \
    && rm /tmp/test.mmd /tmp/test.svg

ENTRYPOINT ["/opt/antora/node_modules/.bin/antora"]
CMD ["--stacktrace", "antora-playbook.yml"]
