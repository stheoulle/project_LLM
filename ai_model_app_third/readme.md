ai_model_app/
│
├── data/
│   ├── dicom_images/            # DICOM triés par patient
│   ├── clinical_data.xlsx       # Métadonnées cliniques
│   └── reports/                 # Rapports cliniques par patient (Excel ou JSON)
│
├── models/
│   ├── cnn_backbones/           # ResNet, EfficientNet, RadImageNet
│   ├── transformers/            # Modèle multimodal fusion
│   └── trained/                 # Modèles sauvegardés
│
├── app/
│   ├── main.py                  # Lancement de l’app
│   ├── pipeline.py              # Étapes dynamiques selon données sélectionnées
│   ├── ui.py                    # Interface CLI ou web
│   └── config.yaml              # Configuration des options
│
├── utils/
│   ├── preproc.py               # Chargement, resizing, normalisation des images
│   ├── fusion.py                # Data fusion + encodeurs texte/tabulaires
│   └── metrics.py               # Calcul des métriques
│
└── requirements.txt


Run with 

```bash
python -m app.main
streamlit run app/ui.py
```

Additional CLI utilities

- Ingest a single PDF and save text under docs/:

```bash
python -m utils.pdf_ingest --input /path/to/DSM5.pdf --docs docs --overwrite
```

- Ingest all PDFs from a folder (recursively):

```bash
python -m utils.pdf_ingest --input /path/to/pdfs_folder --docs docs --recursive
```

- Ingest folder and skip existing outputs:

```bash
python -m utils.pdf_ingest --input /path/to/pdfs_folder --docs docs
```

Streamlit UI commands

- Run the Streamlit web UI (includes chat, training and PDF ingestion):

```bash
cd ai_model_app_third
streamlit run app/ui.py
```

Notes

- After ingestion, the UI will attempt to re-index `docs/` so new documents are searchable by the local retriever.
- Ensure you have `pdfplumber` or `PyPDF2` installed for robust PDF extraction.