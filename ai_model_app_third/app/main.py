from app.pipeline import run_pipeline
from app import llm
import sys

if __name__ == "__main__":
    print("📌 Bienvenue dans AI Model App")
    model_choice = "resnet18"
    modality_choice = "docs_embeddings"
    mri_types = ["MRI", "T2", "STIR", "BLISS", "AX", "SENSE", "NA"]
    selected_mri_types = ["MRI"]

    # Small LLM-based analysis when launching from CLI (uses empty chat history by default)
    try:
        suggestion = llm.analyze_chat_and_suggest([], None)
        print(f"LLM suggestion: {suggestion.get('modality')} — {suggestion.get('assistant_text')}")
    except Exception:
        pass

    # Passe les infos à run_pipeline
    run_pipeline(modality_choice, model_choice, selected_mri_types)

