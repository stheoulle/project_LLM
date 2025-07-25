import streamlit as st
from app.pipeline import run_pipeline
from utils.metrics import plot_metrics

st.set_page_config(page_title="AI Model App", layout="wide")

st.title("🧠 AI Model Training App")
st.markdown("Sélectionnez vos options pour lancer un entraînement de modèle sur vos données multimodales.")

# --- Choix du modèle
model_choice = st.selectbox("🎯 Modèle de base", ["resnet18", "radimagenet", "efficientnet"])

# --- Choix des modalités
modality_choice = st.selectbox(
    "🧬 Données à utiliser",
    ["images", "images+meta", "images+meta+reports"]
)

# --- Bouton de lancement
if st.button("🚀 Lancer l'entraînement"):
    with st.spinner("Exécution du pipeline en cours..."):
        history, results = run_pipeline(modality_choice, model_choice, ui=True)
    
    st.success("Entraînement terminé !")
    
    # Affichage des courbes
    st.subheader("📈 Courbes d'entraînement")
    plot_metrics(history, results, streamlit=True)
