# Breast Cancer DBT CNN Classification – End‑to‑End Architecture

This document proposes a complete, production‑ready architecture to build, train, evaluate, deploy, and monitor a breast cancer **classification** system from Digital Breast Tomosynthesis (DBT) DICOM files. It reuses and extends your existing utilities (image reading, bounding boxes, FROC evaluation) and integrates with your preferred stack (Kubernetes on OVH, Terraform, ArgoCD, GitHub CI, Keycloak).

---

## 1) Problem framing & scope

**Primary task**: Breast‑level or exam‑level **classification** (e.g., malignant vs benign vs normal).
**Optional sub‑tasks**: Lesion **detection** to crop ROIs and/or provide CAD overlays, weak supervision (MIL), and multimodal fusion with clinical metadata & reports.

**Modalities**: DBT volumes (3D), optionally 2D synthesized views if available; clinical CSV metadata; free‑text reports.

**Evaluation**: ROC‑AUC / PR‑AUC for classification; sensitivity at fixed FP/volume using FROC for the detection branch; calibration (ECE/Brier); stratified by view and laterality.

---

## 2) High‑level system diagram

```mermaid
flowchart LR
  subgraph DataLayer[Data Layer]
    PACS[(PACS/DICOM Store)] -- DICOM --> Ingest
    Ingest[Ingestion Service]
    Ingest --> RawStore[(Object Storage: s3://dbt-raw)]
    Meta[(Metadata DB)]
    Ann[(Annotations/Boxes CSVs)]
  end

  subgraph Proc[Preprocessing & Feature Engineering]
    Preproc[Preprocessing Workers (K8s)]
    QC[QC & De-identification]
    Patches[ROI Cropping / Patch Gen]
  end

  subgraph Train[Experimentation & Training]
    Orchestrator[Kubeflow/Airflow]
    Trainer[GPU Training Jobs (PyTorch/TF)]
    Reg[Model Registry (MLflow)]
    Track[Experiment Tracking (MLflow/W&B)]
  end

  subgraph Serve[Serving & Apps]
    API[Inference API (FastAPI + Uvicorn)]
    Auth[Keycloak]
    UI[Clinician UI/Viewer]
    Mon[Monitoring & Drift]
  end

  RawStore --> QC --> Preproc --> Patches --> Orchestrator --> Trainer
  Meta -.-> Preproc
  Ann -.-> Patches
  Trainer --> Reg
  Trainer --> Track
  Reg --> API --> UI
  Auth --> UI
  Auth --> API
  API --> Mon
```

---

## 3) Data layer & governance

**3.1 Ingestion**

* DICOM receive via DICOM router or C‑STORE into **Object Storage (S3‑compatible)**.
* Write DICOM keys (`StudyUID`, `SeriesUID`, `SOPInstanceUID`) into a **Metadata DB** (PostgreSQL or DuckDB/Parquet for analytics), with links to object keys.

**3.3 Ground truth & predictions**

* Store **annotations** (boxes) and **labels** in versioned CSV/Parquet:

  * `boxes.parquet`: columns like `PatientID, StudyUID, View, X, Y, Width, Height, Slice, Label(benign/malignant)`.
  * `exam_labels.parquet`: exam‑level labels (target for classification).
  * `filepaths.parquet`: `StudyUID, View, dicom_uri`.
* Maintain a **Prediction Store** (per model/version) for auditability & FROC/ROC analysis.

---

## 4) Preprocessing pipeline

**4.1 DBT loading**

* Reuse your `dcmread_image` as the canonical loader:

  * Decompress (pylibjpeg), windowing (center/width), laterality check & flip for consistent orientation.
  * Add options for **3D handling**: choose `index` (central slice), MIP (max‑intensity projection), or **learned slice selector**.

**4.2 Normalization & resizing**

* Convert to `float32` in \[0,1], per‑study z‑score optional.
* Resize to model input (e.g., 512×512) with aspect‑ratio handling (pad to square).

**4.3 View standardization**

* Normalize view names (e.g., `L-CC`, `R-CC`, `L-MLO`, `R-MLO`).
* Optional **breast mask** to remove background and standardize intensity histogram.

**4.4 ROI generation (optional)**

* Use detection annotations to **crop patches** around lesions.
* Generate negative patches by hard‑negative mining.
* Persist to `s3://dbt-processed/patches/...` with parquet index for fast sampling.

**4.5 Data splits**

* Patient‑level stratified **train/val/test** with temporal leakage prevention.
* Optionally **k‑fold CV** at patient level.

**4.6 Orchestration**

* Implement as **K8s batch jobs** scheduled by Kubeflow Pipelines/Airflow.
* Artifacts tracked in MLflow (dataset version hash, code commit SHA, params).

---

## 5) Modeling strategies

**5.1 Pure classification (breast/exam)**

* Backbone: **2D CNN** (e.g., EfficientNet‑V2, ConvNeXt) pre‑trained on RadImageNet or ImageNet.
* Input variants:

  1. **Single representative slice** per view (e.g., central slice).
  2. **Slice aggregation**: sample N slices → per‑slice embeddings → temporal pooling (mean/max/attention) → view embedding.
  3. **MIP** or **thin‑slab MIP**.
* View‑level heads → **late fusion** across views (CC/MLO, L/R): concat or attention.
* Output: probability per class (binary/multi‑class).

**5.2 Detection‑assisted classification (recommended)**

* Use detection (from boxes) to localize lesions:

  * Detector: light 2D/3D CNN or reuse FROC matching + NMS.
  * Crop ROIs → classify each ROI with a small CNN; aggregate ROI scores to exam‑level (top‑k, noisy‑OR, attention pooling).
* Pros: better focus on suspicious regions; interpretable heatmaps.

**5.3 Multiple‑Instance Learning (MIL)**

* Treat slices/patches as instances in a bag (exam).
* **Attention‑based MIL** for end‑to‑end weak supervision without explicit detectors.

**5.4 Multimodal fusion (phase 2)**

* Clinical CSV (age, density, prior exams, BIRADS): small MLP.
* Report text: ClinicalBERT/RoBERTa (domain‑tuned) → pooled embedding.
* **Fusion**: concat vision embedding + clinical MLP + text embedding → Transformer/MLP head.

---

## 6) Training pipeline

**6.1 Dataset & loaders**

* PyTorch `Dataset` that reads DICOM via `dcmread_image`, applies transforms (RandAugment/flip/contrast), assembles per‑exam batch.
* Weighted sampling to address class imbalance; focal loss or pos‑weight BCE.

**6.2 Losses & optimization**

* Binary: BCEWithLogits + AUC monitoring; Multi‑class: cross‑entropy.
* Optimizer: AdamW; cosine schedule; mixed precision (AMP) for speed.
* Regularization: dropout, stochastic depth, label smoothing.

**6.3 Callbacks & tracking**

* Log metrics (loss, AUC, sensitivity\@fixed specificity), confusion matrices, calibration (ECE), PR curves to MLflow/W\&B.
* Early stopping on **val AUC**; model checkpointing (best‑AUC, best‑calibrated).

**6.4 Hyperparameter search**

* Optuna/Ray Tune on K8s (GPU pool).
* Search over slice strategy, input size, learning rate, fusion type.

**6.5 Reproducibility**

* Seed everything; track data snapshot version (S3 prefix + manifest) and Git SHA.

---

## 7) Evaluation & reporting

**7.1 Metrics**

* Primary: ROC‑AUC, PR‑AUC, sensitivity at fixed specificity (e.g., 0.90/0.95), calibration (Brier, ECE).
* Subgroup analysis: per‑view, per‑breast, density categories, device vendor.

**7.2 Detection branch**

* Reuse your `_is_tp`, `_froc` to produce **FROC** curves and sensitivity @ 1–4 FP/volume.
* Link detection TPs back to classified exams for error analysis.

**7.3 Cross‑validation**

* Patient‑level k‑fold; aggregate AUC with CI (bootstrap).
* Export a **Model Card** including intended use, data, metrics, calibration, and caveats.

---

## 8) Inference & serving

**8.1 Online API**

* **FastAPI** service (GPU optional) running on K8s:

  * Input: `StudyUID` or DICOM URIs.
  * Pipeline: fetch DICOM → preprocess → (optional detector) → classifier → aggregate → post‑processing (calibration, thresholds) → explanation maps.
  * Output: exam probability, per‑view scores, ROI boxes + scores, Grad‑CAM/Score‑CAM overlays.

**8.2 Batch scoring**

* K8s Jobs to score retrospective cohorts; write predictions to Prediction Store (Parquet) for analytics.

**8.3 Explanations**

* CAMs per view; ROI boxes from detector; attach to a viewer (OHIF/Cornerstone) via overlays.

**8.4 Thresholding & triage**

* Choose operating points (Youden’s J, target specificity).
* Optionally dual‑threshold (high‑confidence auto‑clear, low‑confidence radiologist review).

---

## 9) MLOps on Kubernetes (OVH)

**9.1 Infrastructure as Code**

* **Terraform** provisions: OVH managed K8s, GPU node pool, Object Storage, PostgreSQL, Secrets (Vault), ingress, LBs.

**9.2 GitOps**

* **Helm charts** for each component (ingestion, preprocessing workers, trainer, API, UI, monitoring).
* **ArgoCD** sync from Git repos; per‑env overlays with Kustomize (dev/stage/prod).

**9.3 CI**

* **GitHub Actions**: build/test containers, unit/integration tests, security scans (Trivy), push images, update Helm chart versions.

**9.4 Experiment tracking & registry**

* **MLflow** server on K8s: artifact store → S3; backend → Postgres.
* Register champion/challenger models with signatures and input/output schema.

**9.5 Feature/Prediction store**

* Lightweight: Parquet on S3 with Delta/Apache Iceberg for ACID; or Feast if you need online features.

---

## 10) Security, auth, and audit

* **Keycloak** for authN/authZ of UI and API (OIDC; roles: radiologist, admin, data‑engineer).
* **Network policies** isolate namespaces (data vs training vs serving).
* **PHI protection**: De‑identified datasets only in research/training namespaces.
* **Secrets** via K8s sealed‑secrets or Vault; no PHI in logs.
* **Audit**: request/response logs (hashed IDs), dataset manifests, model lineage (MLflow tags).

---

## 11) Monitoring & quality

**11.1 Application**

* Prometheus/Grafana for API latency, throughput, GPU/CPU, errors.

**11.2 ML Monitoring**

* Data drift on input histograms (intensity, breast area, density).
* Prediction drift (mean score, calibration shift); alert if AUC on rolling labeled batches degrades.

**11.3 Human‑in‑the‑loop**

* Radiologist feedback UI to flag false positives/negatives; stream back as annotations to enrich training data.

---

## 12) Data contracts & schemas

* **filepaths.parquet**: `StudyUID (str)`, `View (str)`, `dicom_uri (str)`
* **exam\_labels.parquet**: `PatientID (str)`, `StudyUID (str)`, `label (int|str)`, `split (train/val/test)`
* **boxes.parquet**: `StudyUID (str)`, `View (str)`, `X,Y,Width,Height (int)`, `Slice (int|float)`, `Label`
* **predictions.parquet**: `model_version (str)`, `StudyUID`, `View`, `X,Y,W,H,Z,Score`, `TP (int)`, `GTID (int)`

All parquet files carry schema version + creation timestamp in metadata.

---

## 13) Reference component choices

* **Training**: PyTorch + timm (EfficientNet/ConvNeXt), AMP, DDP across GPUs.
* **Detection** (optional): YOLOv8‑seg/RetinaNet or light 3D CNN; or your geometric TP matching logic.
* **Text**: ClinicalBERT/RoBERTa; simple tokenizer server for speed.
* **Viewer**: OHIF/Cornerstone web viewer with overlay plugins.
* **Pipelines**: Kubeflow Pipelines or Airflow + Argo Workflows for GPU jobs.
* **Storage**: OVH S3 for objects, Postgres for metadata/MLflow backend, MinIO for local dev.

---

## 14) Minimal viable path (MVP → Phase 2)

**MVP (weeks 1–4)**

1. Ingestion + de‑identification to S3; parquet manifests.
2. Preprocessing job reusing `dcmread_image` (slice strategy + resizing).
3. Baseline 2D CNN (central slice) with view‑fusion; ROC‑AUC benchmark.
4. FastAPI inference + Keycloak‑protected endpoint; batch scoring job.
5. MLflow tracking + basic dashboards.

**Phase 2 (weeks 5–10)**

1. Detection‑assisted classification w/ ROI crops and attention pooling.
2. Multimodal fusion (clinical CSV, optional text).
3. CAM explanations in viewer; human‑in‑the‑loop feedback.
4. Drift monitoring + challenger model canary release via Argo Rollouts.

---

## 15) Example package/module layout

```
repo/
├─ infra/                # Terraform, Helm, ArgoCD apps
├─ services/
│  ├─ ingest/            # DICOM receive, de-id, manifest writer
│  ├─ preprocess/        # workers: dcmread_image, resize, slice select
│  ├─ train/             # training scripts, HPO, MLflow loggers
│  ├─ infer-api/         # FastAPI model server, CAMs, Keycloak OIDC
│  └─ viewer/            # OHIF ui + overlays
├─ libs/
│  ├─ dicomio/           # dcmread_image, windowing, laterality utils
│  ├─ datasets/          # PyTorch Datasets, samplers
│  ├─ models/            # backbones, fusion, MIL, detector
│  ├─ eval/              # ROC/PR, FROC, calibration, bootstrap CI
│  └─ utils/             # config, logging, metrics, viz
├─ data_specs/           # parquet schemas, contracts, manifests
└─ pipelines/            # Kubeflow/Airflow DAGs
```

---

## 16) Interfaces & contracts (selected)

**Preprocess job → Parquet manifest**

* Input: `study_list.parquet` (`StudyUID, Views, dicom_uris[]`).
* Output: `preprocessed.parquet` with columns

  * `StudyUID, View, slice_strategy, height, width, tensor_uri`.

**Infer API**

* `POST /v1/predict { study_uid: str }`
* Response: `{ study_uid, prob_malignant: float, view_scores: {L-CC: float, ...}, rois: [...], cams_uri: ... }`
* OIDC: access token required; roles checked via Keycloak.

---

## 17) Risks & mitigations

* **Class imbalance** → weighted loss, focal loss, re‑sampling.
* **Vendor variability** → histogram matching, domain adaptation, fine‑tune per vendor.
* **Label noise** → MIL/ROI aggregation, label smoothing, robust losses.
* **Data leakage** → strict patient‑level splits, no duplicate studies across folds.
* **Calibration drift** → temperature scaling on val set, recalibration periodically.

---

## 18) Where your current code plugs in

* `dcmread_image` → core of **Preprocessing** (windowing/orientation) and **Online Inference** loaders.
* `read_boxes`, `_is_tp`, `_froc` → used in the **Detection‑assisted** path and for detailed **FROC** evaluation.
* `draw_box` → visualization overlays served by the API/Viewer.

---

## 19) Next steps (actionable)

1. Implement preprocessing job with slice strategy flags and write `preprocessed.parquet` + tensors to S3.
2. Train MVP view‑fusion classifier; log to MLflow; export model with signature.
3. Stand up FastAPI + Keycloak OIDC; serve champion model; add CAMs.
4. Add ROI detector → detection‑assisted classifier; enable FROC reports.
5. Integrate OHIF viewer overlays and feedback loop; add drift alerts.
