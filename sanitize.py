import os
import shutil

# === CONFIGURATION ===
source_dir = "CMB-BRCA_v03_20250509/CMB-BRCA"
target_dir = "Filtered-CMB-BRCA"

# Créer le dossier de destination s'il n'existe pas
os.makedirs(target_dir, exist_ok=True)

copied = 0

# Parcours des dossiers MSB-XXXXX
for patient_folder in os.listdir(source_dir):
    patient_path = os.path.join(source_dir, patient_folder)

    if os.path.isdir(patient_path):
        # Sous-dossiers avec noms datés à vérifier
        for study_folder in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study_folder)

            # Vérifier que c'est un dossier, et que le caractère à l'index 10 est 'M'
            if os.path.isdir(study_path) and len(study_folder) > 11 and study_folder[11] == 'M':
                dst_path = os.path.join(target_dir, study_folder)
                if not os.path.exists(dst_path):
                    shutil.copytree(study_path, dst_path)
                    copied += 1
                    print(f"Copié : {study_folder}")
                else:
                    print(f"Déjà existant : {study_folder}")
            else:
                print(f"Non copié (condition non remplie) : {study_folder}")

print(f"\n✅ Total : {copied} dossiers copiés dans {target_dir}")
