# Breast‑Cancer Multimodal Classifier + Chatbot — Architecture & Implementation Plan

Summary

This document describes the full architecture, components, data layout, MCP tool contracts, infra, security requirements and a phased roadmap to build the breast‑cancer imaging + LLM chatbot application using the current repository as source data. A manifest and backup were created; raw data lives in `data/raw/` and the move manifest is `data/manifest/move_manifest.csv`.

Core components

- Data layer
  - data/raw/ — original dataset files (preserved relative paths)
  - data/manifest/move_manifest.csv — original_path, new_path, size_bytes, md5, moved_at
  - data/processed/ — normalized images, patches, segmentation masks
  - data/tables/ — cleaned CSV/XLSX label & metadata tables
  - storage: DVC + remote object storage (S3/GCS) for large artifacts

- Model services (containerized)
  - preprocess-service: image normalization, patch extraction, caching
  - classifier-service: REST API (predict, predict_batch, health)
  - segmentation-service: masks + explainability (Grad‑CAM, IG)
  - embedder-service: image & text embeddings (CLIP / domain encoder)
  - retrieval-service: vector DB queries (FAISS/Milvus/Weaviate)
  - metadata-service: patient/case queries (Postgres)

- MCP tool layer (LLM-facing)
  - Tools expose typed JSON endpoints that the LLM calls.
  - Core tools: classify_image, get_explanation, retrieve_similar, query_metadata, dataset_stats.
  - Every response includes provenance: model_version, manifest_row, timestamp.

- Conversational layer (LLM)
  - Orchestrates tool calls and RAG: retrieval + tool outputs fused into prompt context.
  - Local LLM for dev (Llama family) or hosted provider for production (with strict logging/privacy).
  - Enforce disclaimers & human-in-loop gating for clinical assertions.

- Frontend (UI)
  - Chat interface + image viewer (zoom/pan, overlays) + similar-case browser + metadata panel.
  - Admin UI for manifest, backups, model versions and audit logs.
  - Prototype options: Gradio/Streamlit for quick demo; React for production UI.

Infra & ops

- Containerization: Docker (GPU images for model services)
- Orchestration: docker-compose (dev) → Kubernetes (prod)
- Vector DB: FAISS (local) / Milvus or Weaviate (prod)
- Metadata DB: Postgres
- Model registry & tracking: MLflow or W&B
- Data versioning: DVC for large files + manifest in git
- Monitoring: Prometheus + Grafana + Sentry

Security & compliance

- Ensure sanitized data before any external transfer (use CMMD_clinicaldata_revision_sanitized.xlsx)
- Encrypt data at rest, restrict access via RBAC, log all PHI access
- Keep backups in restricted storage, manifest stored in repo only (no PHI)
- Design LLM prompts/tools to avoid exposing PHI inside provider logs

MCP tool JSON contracts (examples)

- classify_image
  - Request: { "case_id"?: string, "image_bytes"?: base64, "options"?: {...} }
  - Response: { "label":str, "scores":{class:score}, "confidence":float, "model_version":str, "explanation_url"?:str, "provenance":{ "manifest_row":str, "timestamp":str } }

- retrieve_similar
  - Request: { "case_id"?: string, "embedding"?: [float], "k": int }
  - Response: [{ "case_id":str, "similarity":float, "metadata_snippet":{}, "thumbnail_url":str, "manifest_row":str }]

- query_metadata
  - Request: { "case_id": str }
  - Response: { "patient_hash":str, "study_date":str, "labels":{}, "sanitized_fields":{} }

API design (examples)

- POST /tools/classify_image
- POST /tools/retrieve_similar
- GET /cases/{case_id}
- GET /manifest (read-only)

Data & modeling best practices

- Patient-level splits only (avoid leakage)
- Track label provenance (radiology vs pathology) and uncertainty flags
- Address class imbalance (class weights, focal loss, cautious oversampling)
- Use transfer learning (ResNet/ConvNeXt/ViT + domain finetune) and consider multi-input fusion (image + tabular)
- Evaluate AUROC, sensitivity/recall, specificity, calibration, per-center metrics
- Add explainability (Grad‑CAM/IG) and uncertainty estimates (ensembles, MC dropout)

Phased roadmap (recommended sprints)

- Sprint 0 — Data & infra
  - Validate manifest, move_manifest.csv and backup

- Sprint 1 — Basic image ingestion & training
  - Build preprocessing pipeline (data/processed/): normalization, patch extraction, augmentation
  - Create dataset loaders with patient-level splits and stratification
  - Implement training pipeline (PyTorch/PyTorch Lightning) to train on pictures and produce reproducible checkpoints
  - Evaluate baseline metrics and register model artifacts in model registry (MLflow/W&B)

- Sprint 2 — Chatbot mechanism (MVP)
  - Implement retrieval-augmented generation (RAG) pipeline: compute embeddings, index into vector DB, implement retriever
  - Create a simple LLM sandbox (local model) and prompt templates for conversational flows
  - Integrate retriever + LLM to produce contextual answers (ensure PHI is never sent to external providers)
  - Add safety scaffolding (disclaimers, fallback to human review for clinical assertions)

- Sprint 3 — MCP tooling & LLM integration
  - Implement mcp_tools adapter (FastAPI) exposing classify/retrieve/metadata tools with typed JSON contracts
  - Wire the production LLM orchestration to call MCP tools, include caching and provenance in responses

- Sprint 4 — Frontend & explainability (Streamlit)
  - Build the chat UI prototype in Streamlit with image viewer (zoom/pan), overlay support (Grad-CAM, masks)
  - Display similar cases, metadata panels and links to downloadable explainability artifacts

- Sprint 5 — Validation & operations
  - Shadow/prospective validation, monitoring, and drift detection
  - Harden security, audit logging, RBAC and production infra

Rollback & recovery

- Backup: `backups/datasets_20250818.tar.gz` (created)
- Rollback approaches:
  - Untar backup to repo root to restore originals
  - Use `data/manifest/move_manifest.csv` to programmatically move files back (I can provide a revert script)

Quick commands (examples)

- Dry-run move: python3 scripts/move_datasets.py --roots CMB-BRCA Filtered-CMB-BRCA CMMD --target data/raw --manifest data/manifest/move_manifest.csv --dry-run
- Create backup (example): python3 scripts/move_datasets.py --roots <roots...> --backup backups/datasets_YYYYMMDD.tar.gz
