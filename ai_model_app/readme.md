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