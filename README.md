# Breast Cancer Multimodal Classifier & Chatbot — Project Directory Overview

This repository contains code, notebooks, scripts, and data for building a breast cancer multimodal classifier and retrieval-augmented chatbot. The project integrates image, tabular, and text data, supporting preprocessing, training, explainability, and conversational AI.

## Top-Level Structure

- **ARCHITECTURE_PLAN.md** — Detailed architecture, phased roadmap, tool contracts, and best practices.
- **architecture.png / architecture_fork.png / CDD_2_2.png / CDD_2_3.png / CDD_2.png** — Architecture and confusion matrixes.
- **README.md** — Project overview and instructions.
- **.gitignore** — Git ignore rules.

## Notebooks

- **BD_1_preprocess.ipynb, CDD_CNN_Transformer_*.ipynb, CDD_process.ipynb, breast_cancer_multimodal_classifier.ipynb, CMMD_1_preprocess.ipynb, temp.ipynb** — Jupyter notebooks for data exploration, preprocessing, model training, and experiments.

## Applications

- **ai_model_app/** — First version of the app (data loading, training, UI).
- **ai_model_app_second/** — Second version (improved fusion, metrics, model management).
- **ai_model_app_third/** — Third version (RAG pipeline, LLM integration, Streamlit UI).
  - **app/** — Main application code: pipeline, UI, LLM, RAG modules.
  - **utils/** — Utilities for PDF ingestion, preprocessing, metrics.
  - **scripts/** — CLI scripts for RAG example generation, validation, fine-tuning.
  - **docs/** — Text documents for retrieval and training (e.g., WHO guidelines).
  - **ai_model_app_second/readme.md** — Usage instructions and CLI commands.

## Scripts

- **scripts/move_datasets.py** — Move and manifest dataset files, supports backup and rollback.
- **0_classify_chat/scripts/prepare_5pct.py** — Sample 5% of images for baseline training.
- **ai_model_app_third/scripts/auto_generate_rag_examples.py** — Auto-generate RAG training examples from docs.
- **ai_model_app_third/scripts/validate_rag_examples.py** — Validate and clean RAG training examples.
- **ai_model_app_third/scripts/finetune_t5_rag.py** — Fine-tune T5 model on RAG examples.

## Other Components

- **requirements.txt** — Python dependencies for each app version.
- **model files (.h5, .npy, .npz)** — Saved model weights and embeddings.
- **image directories** — CDD-CESM image folders (low energy, subtracted), JSON outputs, processed metadata.

## CNN Scripts

The repository includes several scripts and notebooks focused on convolutional neural network (CNN) architectures for breast cancer image classification. These scripts typically handle image preprocessing, model definition, training, evaluation, and visualization of results. Key files include:

- **CDD_*.ipynb** — Notebooks that implement and compare CNN and transformer-based models for multimodal data.
- **CDD_CNN_Transformer_*.ipynb** — Notebooks that implement and compare CNN and transformer-based models for multimodal data.
- **CDD_process.ipynb** — Preprocessing and pipeline setup for CNN-based experiments.
- **ai_model_app/app/cnn_model.py** (if present) — Contains reusable CNN model classes and training routines.
- **scripts/move_datasets.py** — Supports organizing image datasets for CNN input.

These scripts are designed to work with medical imaging data, leveraging CNNs for feature extraction and classification tasks. They may also include explainability modules to interpret model predictions.

---
