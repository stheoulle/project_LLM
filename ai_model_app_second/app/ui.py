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
    [
        "cbis_cases",           # ✅ New: CBIS-DDSM case-level images + CSV metadata
        "images",
        "images+meta",
        "images+meta+reports",
    ]
)

# --- Hyperparameters & options
with st.sidebar:
    st.header("⚙️ Paramètres d'entraînement")
    epochs_img = st.number_input("Epochs (image)", min_value=1, max_value=100, value=10, step=1)
    epochs_fusion = st.number_input("Epochs (fusion)", min_value=1, max_value=100, value=10, step=1)
    batch_size = st.number_input("Batch size", min_value=4, max_value=128, value=32, step=4)
    patience_img = st.number_input("Patience (image)", min_value=1, max_value=20, value=3, step=1)
    patience_fusion = st.number_input("Patience (fusion)", min_value=1, max_value=20, value=5, step=1)
    use_balanced_sampler = st.checkbox("Balanced sampler (inverse fréquence)", value=True)
    val_size = st.slider("Validation size", min_value=0.05, max_value=0.4, value=0.2, step=0.05)

# --- Bouton de lancement
if st.button("🚀 Lancer l'entraînement"):
    with st.spinner("Exécution du pipeline en cours..."):
        history, results = run_pipeline(
            modality_choice,
            model_choice,
            ui=True,
            epochs_img=epochs_img,
            epochs_fusion=epochs_fusion,
            batch_size=batch_size,
            patience_img=patience_img,
            patience_fusion=patience_fusion,
            use_balanced_sampler=use_balanced_sampler,
            val_size=val_size,
        )
    
    st.success("Entraînement terminé !")
    
    # Affichage des courbes
    st.subheader("📈 Courbes d'entraînement")
    plot_metrics(history, results, streamlit=True)
