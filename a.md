
❯ rebuild docker comspoe and start                                                                                      
                                   
⏺ Bash(docker compose up --build -d 2>&1)                                                                               
  ⎿   Image postgres:16-alpine Pulling                                                                              
      Image postgres:16-alpine Pulled                                                                                   
      Image lila-backend Building                                                                                       
     … +199 lines (ctrl+o to expand)                                                                                    
  ⎿  (timeout 5m)                                                                                                       
                                                                                                                        
⏺ All containers are up and running:                                                                                    
                                                                                                                        
  - Dashboard: http://localhost:3000                                                                                    
  - Backend API: http://localhost:8000                                                                                  
  - Docs: http://localhost:8081                                                                                         
  - Ollama: http://localhost:11434                                                                                      
  - ollama-init is pulling nomic-embed-text and phi3:mini in the background                                             

✻ Brewed for 33s                        
                                                                                                                        
❯ describe pipeline in as arcticle, use mermaid for data flow and data processing state flow                            
                                                                                             
⏺ LALI — Local AI Log Intelligence: Pipeline Architecture                                                               
                                         
  Overview                                                                                                              
                                                            
  LALI is a self-hosted log intelligence platform that transforms raw Kibana CSV exports into actionable insights       
  through a multi-stage pipeline combining deterministic analysis with LLM-powered reasoning. Every component runs
  locally — no data leaves the infrastructure.

  ---
  Data Flow

  flowchart LR
      CSV["Kibana CSV Export"]
      subgraph Ingestion
          Parser["Parser"]
          Masker["Sensitive Data Masker"]
      end
      subgraph Storage
          LR_DB[("PostgreSQL")]
          LogRecords["LogRecords"]
          Sessions["SessionRecords"]
          Embeddings["LogEmbeddings"]
      end
      subgraph Analysis Engine
          Classification["Classification"]
          Clustering["Error Clustering"]
          Frequency["Frequency Analysis"]
          Anomaly["Anomaly Detection\n(Z-Score)"]
      end
      subgraph LLM Layer
          Ollama["Ollama"]
          Summary["Summary Generation"]
          Explain["Anomaly Explanation"]
          RootCause["Root Cause Analysis"]
          Embed["Semantic Embedding"]
          SemanticAnom["Semantic Anomaly\nDetection"]
      end
      subgraph Presentation
          API["FastAPI REST API"]
          Dashboard["React Dashboard"]
      end

      CSV --> Parser --> Masker --> LogRecords --> LR_DB
      LogRecords --> Sessions --> LR_DB

      LR_DB --> Classification
      LR_DB --> Clustering
      LR_DB --> Frequency
      LR_DB --> Anomaly

      LR_DB --> Embed --> Embeddings --> LR_DB
      Embeddings --> SemanticAnom

      Classification --> Summary
      Clustering --> Summary
      Clustering --> RootCause
      Frequency --> Summary
      Anomaly --> Explain

      Summary --> Ollama
      Explain --> Ollama
      RootCause --> Ollama
      Embed --> Ollama

      Classification --> API
      Clustering --> API
      Frequency --> API
      Anomaly --> API
      Summary --> API
      Explain --> API
      RootCause --> API
      SemanticAnom --> API

      API --> Dashboard

  ---
  Processing State Flow

  Each log record moves through a deterministic state machine from raw text to enriched, analysed data:

  stateDiagram-v2
      [*] --> Uploaded: CSV file received

      Uploaded --> Parsing: parse_csv()
      Parsing --> Masked: mask_message()
      Masked --> Structured: _parse_message()

      state Structured {
          [*] --> FieldExtraction
          FieldExtraction --> TypeClassified: log_type assigned
          TypeClassified --> TimestampParsed: _parse_timestamp()
      }

      Structured --> Stored: store_logs()
      Stored --> SessionBuilt: build_sessions()

      state SessionBuilt {
          [*] --> Grouped: group by tracking_id
          Grouped --> Paired: match request ↔ response
          Paired --> Enriched: compute duration, is_error
      }

      SessionBuilt --> ReadyForAnalysis

      state ReadyForAnalysis {
          [*] --> StatisticalAnalysis
          [*] --> LLMAnalysis

          state StatisticalAnalysis {
              [*] --> Classified: status + error type
              [*] --> Clustered: error code grouping
              [*] --> FrequencyAnalysed: time windows + spikes
              [*] --> AnomalyDetected: z-score on errors, rate, volume
          }

          state LLMAnalysis {
              [*] --> Embedded: nomic-embed-text → 768-dim vectors
              Embedded --> KMeansClustered: k=10, 20 iterations
              KMeansClustered --> SemanticAnomaliesFound: rank by centroid distance
              [*] --> RootCausesIdentified: error sessions → LLM grouping
              AnomalyDetected --> AnomalyExplained: LLM narrative per window
              Classified --> SummaryGenerated: LLM synthesis
          }
      }

      ReadyForAnalysis --> Served: REST API
      Served --> Visualised: React Dashboard
      Visualised --> [*]

  ---
  Pipeline Stages in Detail

  Stage 1 — Ingestion

  The entry point accepts either a file upload (POST /api/v1/logs/upload) or a server-side path (POST
  /api/v1/logs/ingest). The parser handles Kibana's specific CSV dialect: comma-separated with a timestamp column in
  "Feb 9, 2026 @ 13:03:50.657" format and a freeform message body.

  Before any data is stored, the masking layer applies 10 regex patterns to redact authorization headers, bearer tokens,
   cookies, session IDs, API keys, and passwords. The original text is never persisted — only the masked variant reaches
   the database.

  Each message is then classified by type (request_incoming, response_outgoing, or unknown) and its structured fields —
  URI, HTTP method, status code, payload presence — are extracted via key-value parsing of the message body.

  Stage 2 — Session Construction

  Raw log records are grouped by tracking_id to reconstruct request-response sessions. The system pairs each incoming
  request with its outgoing response, computes round-trip duration in milliseconds, and flags sessions as errors when
  the response status is ≥ 400. This produces the SessionRecord table that serves as the foundation for all downstream
  analysis.

  Stage 3 — Statistical Analysis

  Four deterministic analysers run against the stored data:

  ┌─────────────────┬───────────────────────────────────────────────────────────┬───────────────────────────────────┐
  │    Analyser     │                          Method                           │              Output               │
  ├─────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────┤
  │                 │ Maps status codes to categories; regex-matches error      │ Error rate, status distribution,  │
  │ Classification  │ payloads against 9 known patterns (timeout, auth failure, │ error type distribution           │
  │                 │  upstream error, etc.)                                    │                                   │
  ├─────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────┤
  │ Error           │ Extracts code field from JSON response payloads; groups   │ Ranked clusters with counts and   │
  │ Clustering      │ by error code                                             │ sample tracking IDs               │
  ├─────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────┤
  │ Frequency       │ Buckets all responses into configurable time windows;     │ Time series, spike list, top      │
  │ Analysis        │ detects spikes where error rate exceeds 2× the overall    │ errors by status and code         │
  │                 │ average                                                   │                                   │
  ├─────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────┤
  │ Anomaly         │ Computes z-scores on three metrics (error count, error    │ Anomaly list with z-scores,       │
  │ Detection       │ rate, request volume) per time window; flags windows      │ severity (warning / critical),    │
  │                 │ exceeding the threshold                                   │ and stats                         │
  └─────────────────┴───────────────────────────────────────────────────────────┴───────────────────────────────────┘

  Stage 4 — LLM-Powered Intelligence

  This layer uses a locally-hosted Ollama instance to add reasoning capabilities that go beyond pattern matching. Every
  LLM feature includes a rule-based fallback for environments where Ollama is unavailable.

  Semantic Embedding & Anomaly Detection — Log messages are embedded into 768-dimensional vectors via nomic-embed-text
  through Ollama's /api/embed endpoint. A pure-Python k-means implementation (k=10, 20 iterations, seeded for
  reproducibility) clusters these vectors. Logs with the greatest cosine distance to their assigned centroid are
  surfaced as semantic anomalies — patterns the system has never seen before, regardless of HTTP status code.

  Root Cause Analysis — Error sessions are collected, serialised as JSON, and sent to the LLM with instructions to group
   them into named root causes with severity ratings and explanations. The fallback path uses the existing error
  clustering logic and assigns severity based on occurrence count thresholds (≥10 critical, ≥5 high, ≥2 medium).

  Anomaly Explanation — For any detected statistical anomaly, the system retrieves the logs in that time window and asks
   the LLM to produce a 2–4 sentence narrative explaining the likely cause. The fallback constructs a factual summary
  citing the top error status and affected endpoints.

  Summary Generation — Combines classification, clustering, and frequency data into a structured prompt. The LLM returns
   a 3–5 sentence health assessment covering system status, critical issues, and recommended actions.

  Stage 5 — Presentation

  The FastAPI backend exposes 11 REST endpoints under /api/v1/. The React dashboard consumes these through a typed API
  client and renders:

  - Overview — health metrics, status distribution pie charts, top error clusters
  - Errors — error cluster table with sample tracking IDs
  - Time Series — request volume bar chart, error rate line chart, anomaly markers with per-row "Explain" buttons that
  expand to show LLM narratives
  - Sessions — filterable session list with drill-down to individual request/response logs
  - Semantic Anomalies — embed trigger button, anomaly table ranked by centroid distance
  - Root Causes — severity-coded cards with explanations and sample tracking IDs

  ---
  Infrastructure

  flowchart TB
      subgraph Docker Compose
          PG["PostgreSQL 16"]
          OL["Ollama Server"]
          OI["ollama-init\n(pulls nomic-embed-text\n+ phi3:mini)"]
          BE["FastAPI Backend\n:8000"]
          FE["React Dashboard\n:3000"]
          DO["Docs Site\n:8081"]
      end

      OI -->|depends_on| OL
      BE -->|depends_on| PG
      BE -->|depends_on| OL
      FE -->|depends_on| BE

      BE -- "DATABASE_URL" --> PG
      BE -- "OLLAMA_BASE_URL\nOLLAMA_MODEL\nOLLAMA_EMBED_MODEL" --> OL

  All models run on the Ollama instance — phi3:mini for text generation, nomic-embed-text for embeddings. The
  ollama-init sidecar container pulls both models on first startup so the system is ready without manual intervention.
