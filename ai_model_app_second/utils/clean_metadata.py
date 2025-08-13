import pandas as pd
import re

# Charger le fichier (remplace le nom de fichier par le tien)
input_file = "Breast-diagnosis/TCIA-Breast-clinical-data-public-7_16_11.xlsx"  # ou .csv
sheet_name = 0  # ou le nom de la feuille Excel

# Fonction de nettoyage et de normalisation
def normalize_pathology(raw_text):
    text = str(raw_text).lower()
    text = re.sub(r"[^\w\s]", "", text)  # enlever ponctuation
    text = re.sub(r"\s+", " ", text).strip()

    # Mapping des pathologies vers une forme standardisée
    mapping = {
        "benign fibrosis": "Benign Fibrosis",
        "benign fibroadenoma": "Benign Fibroadenoma",
        "benign fibrocyst": "Benign Fibrocyst",
        "benign": "Benign",
        "stromal hyperplasia no ca": "Stromal Hyperplasia",
        "infiltrat lobular": "Lobular Infiltrate",
        "invasive ductal ca": "Invasive Ductal Carcinoma",
        "invasive ductal carcinoma": "Invasive Ductal Carcinoma",
        "infiltrating ductal ca": "Infiltrating Ductal Carcinoma",
        "invasive ductal": "Invasive Ductal Carcinoma",
        "invasive intraductal": "Invasive Intraductal Carcinoma",
        "invasive intraductal ca": "Invasive Intraductal Carcinoma",
        "ductal ca in situ": "Ductal Carcinoma In Situ",
        "ductal carcinoma in situ": "Ductal Carcinoma In Situ",
        "ductal carcinoma insitu": "Ductal Carcinoma In Situ",
        "ductal carcinoma in-situ": "Ductal Carcinoma In Situ",
        "intraductal ca": "Intraductal Carcinoma",
        "invasive lobular": "Invasive Lobular Carcinoma",
        "invasive lobular carcinoma": "Invasive Lobular Carcinoma",
        "invasive carcinoma": "Invasive Carcinoma",
    }

    # Recherche du mapping
    for key in mapping:
        if key in text:
            return mapping[key]

    # Si non reconnu
    return "Unknown"

# Lire le fichier
if input_file.endswith(".xlsx"):
    df = pd.read_excel(input_file, sheet_name=sheet_name)
else:
    df = pd.read_csv(input_file)

# Assure-toi que la colonne existe
if df.shape[1] < 3:
    raise ValueError("Le fichier doit avoir au moins 3 colonnes.")

# Normalisation de la 3e colonne
df["normalized_pathology"] = df.iloc[:, 2].apply(normalize_pathology)

# Sauvegarder le résultat
output_file = input_file.replace(".xlsx", "_normalized.csv").replace(".csv", "_normalized.csv")
df.to_csv(output_file, index=False)

print(f"Fichier sauvegardé : {output_file}")
