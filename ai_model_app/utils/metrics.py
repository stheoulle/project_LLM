import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import torch
import torch.nn.functional as F
import seaborn as sns

# 🔧 Redimensionne un volume 3D à une profondeur fixe (D)
def resize_volume(volume, target_depth=128):
    # volume: [C, D, H, W]
    volume = volume.unsqueeze(0)  # [1, C, D, H, W]
    resized = F.interpolate(volume, size=(target_depth, volume.shape[3], volume.shape[4]),
                            mode='trilinear', align_corners=False)
    return resized.squeeze(0)  # [C, D, H, W]

# 🧪 Évalue le modèle après avoir redimensionné les images
def evaluate_model(model, data, target_depth=128):
    model.eval()

    # Resize all volumes to fixed depth
    resized_images = [resize_volume(img, target_depth) for img in data['images']]
    inputs = torch.stack(resized_images)
    labels = data['labels']

    with torch.no_grad():
        preds = model(inputs).argmax(1)

    report = classification_report(labels.cpu(), preds.cpu(), output_dict=True)
    return {"y_true": labels, "y_pred": preds, "report": report}

def plot_metrics(history, results, streamlit=False):
    # 📈 Courbes de perte et d'accuracy
    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Loss')
    ax.plot(history['acc'], label='Accuracy')
    ax.set_title("Courbes d’entraînement")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valeur")
    ax.legend()

    # 📊 Matrice de confusion
    y_true = results['y_true'].cpu().numpy()
    y_pred = results['y_pred'].cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    cm_fig, cm_ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=cm_ax)
    cm_ax.set_title("Matrice de confusion")
    cm_ax.set_xlabel("Prédiction")
    cm_ax.set_ylabel("Réel")

    if streamlit:
        st.pyplot(fig)
        st.pyplot(cm_fig)
        st.json(results['report'])
    else:
        plt.show()  # courbes + confusion matrix
        print("📊 Rapport de classification :")
        for label, metrics in results['report'].items():
            if isinstance(metrics, dict):
                print(f"{label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}")
