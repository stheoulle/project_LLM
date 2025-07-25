from app.pipeline import run_pipeline
import sys

if __name__ == "__main__":
    print("📌 Bienvenue dans AI Model App")
    #model_choice = input("Choisissez un modèle (resnet18 / radimagenet / efficientnet): ").strip().lower()
    model_choice = "resnet18"  # Pour simplifier, on utilise un modèle par défaut
    #modality_choice = input("Choisissez les modalités (images / images+meta / images+meta+reports): ").strip().lower()
    modality_choice = "images"  # Pour simplifier, on utilise une modalité par défaut
    mri_types = ["MRI", "T2", "STIR", "BLISS", "AX", "SENSE", "NA"]
    selected_mri_types = []

    # Si le choix contient "images", on demande quel(s) type(s) de MRI
    if "images" in modality_choice:
    #     print("Choisissez un ou plusieurs types de MRI parmi :")
    #     print(", ".join(mri_types))
    #     mri_input = input("Tapez les types séparés par des virgules (ex: MRI,T2,AX): ").strip()
        # selected_mri_types = [mri.strip().upper() for mri in mri_input.split(",") if mri.strip().upper() in mri_types]
        selected_mri_types = ["MRI"]  # Pour simplifier, on utilise un type par défaut
        if not selected_mri_types:
            print("Aucun type de MRI valide sélectionné, utilisation par défaut : ['MRI']")
            selected_mri_types = ["MRI"]

    # Passe les infos à run_pipeline
    run_pipeline(modality_choice, model_choice, selected_mri_types)

